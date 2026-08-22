import Foundation

public struct ThermalObservation: Codable, Hashable, Sendable {
  public let atEpochMilliseconds: Int64
  public let level: Int
  public let rawValue: String
}

public struct ThermalTraceSummary: Codable, Sendable {
  public let source: String
  public let command: String
  public let rawTraceSha256: String
  public let startedAtEpochMilliseconds: Int64
  public let completedAtEpochMilliseconds: Int64
  public let observationCount: Int
  public let maximumGapMilliseconds: Int64
  public let nonNominalCount: Int
  public let observations: [ThermalObservation]
}

public struct ThermalGateEvidence: Codable, Sendable {
  public let source: String
  public let command: String
  public let startedAtEpochMilliseconds: Int64
  public let completedAtEpochMilliseconds: Int64
  public let observationCount: Int
  public let maximumGapMilliseconds: Int64
  public let nonNominalCount: Int
  public let observations: [ThermalObservation]

  public init(observations: [ThermalObservation]) throws {
    guard let first = observations.first, let last = observations.last else {
      throw Opt0068Error.contract("thermal gate contains no observations")
    }
    var maximumGap: Int64 = 0
    for pair in zip(observations, observations.dropFirst()) {
      maximumGap = max(maximumGap, pair.1.atEpochMilliseconds - pair.0.atEpochMilliseconds)
    }
    let nonNominal = observations.filter { $0.level != 0 }.count
    guard nonNominal == 0,
          last.atEpochMilliseconds - first.atEpochMilliseconds >= 30_000,
          maximumGap <= Opt0068Frozen.maximumThermalPollGapMilliseconds else {
      throw Opt0068Error.contract("thermal gate evidence is not continuous nominal state")
    }
    self.source = ThermalMonitor.source
    self.command = ThermalMonitor.command
    self.startedAtEpochMilliseconds = first.atEpochMilliseconds
    self.completedAtEpochMilliseconds = last.atEpochMilliseconds
    self.observationCount = observations.count
    self.maximumGapMilliseconds = maximumGap
    self.nonNominalCount = nonNominal
    self.observations = observations
  }
}

public final class ThermalMonitor: @unchecked Sendable {
  public static let source = "macOS notifyutil thermal pressure"
  public static let command = "/usr/bin/notifyutil -g com.apple.system.thermalpressurelevel"

  private let traceURL: URL
  private let lock = NSLock()
  private var observations: [ThermalObservation] = []
  private var shouldStop = false
  private var thread: Thread?
  private var terminalError: Error?

  public init(traceURL: URL) {
    self.traceURL = traceURL
  }

  public func start() throws {
    lock.lock()
    defer { lock.unlock() }
    guard thread == nil else { throw Opt0068Error.contract("thermal monitor already started") }
    guard !FileManager.default.fileExists(atPath: traceURL.path) else {
      throw Opt0068Error.contract("thermal trace already exists: \(traceURL.path)")
    }
    try FileManager.default.createDirectory(
      at: traceURL.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    FileManager.default.createFile(atPath: traceURL.path, contents: nil)
    let worker = Thread { [weak self] in self?.pollLoop() }
    worker.name = "ace-opt-0068-thermal"
    thread = worker
    worker.start()
  }

  public func awaitNominal(seconds: Double, timeoutSeconds: Double = 900) throws -> [ThermalObservation] {
    let deadline = Date().addingTimeInterval(timeoutSeconds)
    var nominalStart: Int64?
    while Date() < deadline {
      if let error = snapshotError() { throw error }
      let snapshot = currentObservations()
      if let latest = snapshot.last {
        if snapshot.count > 1 {
          let gap = latest.atEpochMilliseconds - snapshot[snapshot.count - 2].atEpochMilliseconds
          if gap > Opt0068Frozen.maximumThermalPollGapMilliseconds { nominalStart = nil }
        }
        if latest.level == 0 {
          if nominalStart == nil { nominalStart = latest.atEpochMilliseconds }
          if Double(latest.atEpochMilliseconds - nominalStart!) >= seconds * 1_000 {
            return snapshot.filter { $0.atEpochMilliseconds >= nominalStart! }
          }
        } else {
          nominalStart = nil
        }
      }
      Thread.sleep(forTimeInterval: 0.1)
    }
    throw Opt0068Error.contract("thermal level 0 gate timed out without \(seconds) continuous seconds")
  }

  public func stop() throws -> ThermalTraceSummary {
    lock.lock()
    shouldStop = true
    let worker = thread
    lock.unlock()
    while worker?.isFinished == false { Thread.sleep(forTimeInterval: 0.05) }
    if let error = snapshotError() { throw error }
    let snapshot = currentObservations()
    guard let first = snapshot.first, let last = snapshot.last else {
      throw Opt0068Error.contract("thermal trace contains no observations")
    }
    var maximumGap: Int64 = 0
    for pair in zip(snapshot, snapshot.dropFirst()) {
      maximumGap = max(maximumGap, pair.1.atEpochMilliseconds - pair.0.atEpochMilliseconds)
    }
    return ThermalTraceSummary(
      source: Self.source,
      command: Self.command,
      rawTraceSha256: try Opt0068Contract.sha256(file: traceURL),
      startedAtEpochMilliseconds: first.atEpochMilliseconds,
      completedAtEpochMilliseconds: last.atEpochMilliseconds,
      observationCount: snapshot.count,
      maximumGapMilliseconds: maximumGap,
      nonNominalCount: snapshot.filter { $0.level != 0 }.count,
      observations: snapshot
    )
  }

  public func awaitObservation(afterEpochMilliseconds timestamp: Int64) throws {
    let deadline = Date().addingTimeInterval(5)
    while Date() < deadline {
      if let error = snapshotError() { throw error }
      if currentObservations().last?.atEpochMilliseconds ?? 0 > timestamp { return }
      Thread.sleep(forTimeInterval: 0.05)
    }
    throw Opt0068Error.contract("thermal monitor did not observe native-owner cleanup")
  }

  private func pollLoop() {
    do {
      let handle = try FileHandle(forWritingTo: traceURL)
      defer { try? handle.close() }
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
      while true {
        lock.lock()
        let stop = shouldStop
        lock.unlock()
        if stop { break }
        let raw = try Self.readRawThermalValue()
        guard let level = Int(raw.split(whereSeparator: \.isWhitespace).last ?? "") else {
          throw Opt0068Error.contract("notifyutil returned unparseable thermal value: \(raw)")
        }
        let observation = ThermalObservation(
          atEpochMilliseconds: Int64((Date().timeIntervalSince1970 * 1_000).rounded()),
          level: level,
          rawValue: raw
        )
        var line = try encoder.encode(observation)
        line.append(0x0a)
        try handle.write(contentsOf: line)
        try handle.synchronize()
        lock.lock()
        observations.append(observation)
        lock.unlock()
        Thread.sleep(forTimeInterval: Double(Opt0068Frozen.thermalPollMilliseconds) / 1_000)
      }
    } catch {
      lock.lock()
      terminalError = error
      lock.unlock()
    }
  }

  private static func readRawThermalValue() throws -> String {
    let process = Process()
    let pipe = Pipe()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/notifyutil")
    process.arguments = ["-g", "com.apple.system.thermalpressurelevel"]
    process.standardOutput = pipe
    process.standardError = pipe
    try process.run()
    process.waitUntilExit()
    let raw = String(
      data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8
    )?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    guard process.terminationStatus == 0 else {
      throw Opt0068Error.contract("notifyutil failed (\(process.terminationStatus)): \(raw)")
    }
    return raw
  }

  private func currentObservations() -> [ThermalObservation] {
    lock.lock()
    defer { lock.unlock() }
    return observations
  }

  private func snapshotError() -> Error? {
    lock.lock()
    defer { lock.unlock() }
    return terminalError
  }
}

public enum Opt0068Process {
  public static func output(_ executable: String, _ arguments: [String]) throws -> String {
    let process = Process()
    let pipe = Pipe()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments
    process.standardOutput = pipe
    process.standardError = pipe
    try process.run()
    process.waitUntilExit()
    let result = String(
      data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8
    )?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    guard process.terminationStatus == 0 else {
      throw Opt0068Error.contract("\(executable) failed: \(result)")
    }
    return result
  }
}
