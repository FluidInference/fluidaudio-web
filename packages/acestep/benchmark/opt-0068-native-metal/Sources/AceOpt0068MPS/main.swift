import AceOpt0068Core
import Accelerate
import CryptoKit
import Foundation
import Metal
import MetalPerformanceShaders

private struct Arguments {
  enum Mode: String { case describe, inspect, correctness, measure, sustained }

  var mode: Mode = .describe
  var packageDirectory: URL?
  var fixtureManifest: URL?
  var output: URL?
  var thermalTrace: URL?
  var harnessCommit: String?
  var harnessSourceRoot: URL?
  var consent: String?
  var samples = 5
  var warmups = 2
  var fullCPUContract = false

  static func parse() throws -> Self {
    var result = Self()
    var iterator = CommandLine.arguments.dropFirst().makeIterator()
    while let argument = iterator.next() {
      func value() throws -> String {
        guard let value = iterator.next() else {
          throw Opt0068Error.contract("missing value after \(argument)")
        }
        return value
      }
      switch argument {
      case "--mode":
        guard let mode = Mode(rawValue: try value()) else {
          throw Opt0068Error.contract("mode must be describe, inspect, correctness, measure, or sustained")
        }
        result.mode = mode
      case "--package-dir": result.packageDirectory = URL(fileURLWithPath: try value())
      case "--fixture-manifest": result.fixtureManifest = URL(fileURLWithPath: try value())
      case "--output": result.output = URL(fileURLWithPath: try value())
      case "--thermal-trace": result.thermalTrace = URL(fileURLWithPath: try value())
      case "--harness-commit": result.harnessCommit = try value()
      case "--harness-source-root": result.harnessSourceRoot = URL(fileURLWithPath: try value())
      case "--execute-native-gpu": result.consent = try value()
      case "--samples": result.samples = Int(try value()) ?? 0
      case "--warmups": result.warmups = Int(try value()) ?? -1
      case "--full-cpu-contract": result.fullCPUContract = true
      default: throw Opt0068Error.contract("unknown argument \(argument)")
      }
    }
    return result
  }
}

private struct StageTiming: Codable {
  let executed: Bool
  let reason: String?
  let encodingMilliseconds: Double
  let commitThroughFenceMilliseconds: Double
  let gpuMilliseconds: Double
  let synchronizationOverheadMilliseconds: Double
  let commandThroughFenceMilliseconds: Double
}

private struct CombinedSample: Codable {
  let caseId: String
  let repetition: Int
  let startedAtEpochMilliseconds: Int64
  let completedAtEpochMilliseconds: Int64
  let flops: UInt64
  let encodingMilliseconds: Double
  let commitThroughFenceMilliseconds: Double
  let gpuMilliseconds: Double
  let synchronizationOverheadMilliseconds: Double
  let commandThroughFenceMilliseconds: Double
  let tflopsIncludingEpilogueAndMaterialization: Double
  let outputSha256: String
  let canaryPassed: Bool
}

private struct StageProfile: Codable {
  let caseId: String
  let contraction: StageTiming
  let biasEpilogue: StageTiming
  let materialization: StageTiming
  let biasMode: String
  let outputBoundary: String
}

private struct CorrectnessCase: Codable {
  let id: String
  let candidateVersusAcceptedWebGPU: NumericMetrics
  let acceptedWebGPUVersusCPU: NumericMetrics?
  let candidateVersusCPU: NumericMetrics?
  let candidateOutputSha256: String
  let repeatedOutputSha256: String
  let deterministic: Bool
  let canaryPassed: Bool
  let passed: Bool
}

private struct AdversarialCase: Codable {
  let id: String
  let metrics: NumericMetrics
  let deterministic: Bool
  let canaryPassed: Bool
  let accumulationMagnitudePassed: Bool
  let passed: Bool
}

private struct MachineIdentity: Codable {
  let model: String
  let memoryBytes: UInt64
  let osVersion: String
  let osBuild: String
  let metalDevice: String
  let gpuCoreCount: Int
  let metalFamilies: [String]
  let mpsFrameworkVersion: String
  let mpsFrameworkSDK: String
  let swiftVersion: String
  let sdkPath: String
  let executableSha256: String
  let sourceBundleSha256: String
  let harnessCommit: String
}

private struct PerformanceSummary: Codable {
  let samples: [CombinedSample]
  let stageProfiles: [StageProfile]
  let weightedWallMillisecondsByRepetition: [Double]
  let weightedTFLOPSByRepetition: [Double]
  let medianWeightedWallMilliseconds: Double
  let medianWeightedTFLOPS: Double
  let denseRateGatePassed: Bool
  let webGPUComparisonPerformed: Bool
  let pairedWebGPUSpeedups: [Double]?
  let denseSpeedupGatePassed: Bool?
  let completePhaseOneDenseGatePassed: Bool
  let sustainedIntervals: [SustainedInterval]?
  let sustainedRetention: Double?
  let sustainedRetentionGatePassed: Bool?
}

private struct SustainedInterval: Codable {
  let index: Int
  let startedMilliseconds: Double
  let completedMilliseconds: Double
  let completedFlops: UInt64
  let tflops: Double
}

private struct Receipt: Codable {
  let schema: String
  let experimentId: String
  let status: String
  let disposition: String
  let claimBoundary: [String]
  let identity: MachineIdentity
  let packageManifestSha256: String
  let activationFixtureManifestSha256: String
  let arithmetic: [String: String]
  let correctness: [CorrectnessCase]
  let adversarial: [AdversarialCase]
  let performance: PerformanceSummary?
  let thermalGate: ThermalGateEvidence?
  let thermal: ThermalTraceSummary?
  let fullCPUContractExecuted: Bool
  let allMandatoryScreensPassed: Bool
}

private final class PreparedCase {
  let resolved: ResolvedCase?
  let id: String
  let rows: Int
  let inner: Int
  let columns: Int
  let flops: UInt64
  let expected: [Float]
  let activation: [Float16]
  let weight: [Float16]
  let left: MPSMatrix
  let right: MPSMatrix
  let result: MPSMatrix
  let resultBuffer: MTLBuffer
  let readback: MTLBuffer
  let multiplication: MPSMatrixMultiplication

  init(
    device: MTLDevice,
    queue: MTLCommandQueue,
    resolved: ResolvedCase?,
    id: String,
    rows: Int,
    inner: Int,
    columns: Int,
    activation: [Float16],
    weight: [Float16],
    expected: [Float]
  ) throws {
    self.resolved = resolved
    self.id = id
    self.rows = rows
    self.inner = inner
    self.columns = columns
    self.flops = UInt64(2) * UInt64(rows) * UInt64(inner) * UInt64(columns)
    self.activation = activation
    self.weight = weight
    self.expected = expected
    let aBytes = activation.count * MemoryLayout<Float16>.stride
    let bBytes = weight.count * MemoryLayout<Float16>.stride
    let c32Bytes = rows * columns * MemoryLayout<Float>.stride
    guard let a = Self.privateBuffer(device: device, queue: queue, values: activation),
          let b = Self.privateBuffer(device: device, queue: queue, values: weight),
          let c32 = device.makeBuffer(length: c32Bytes, options: .storageModePrivate),
          let readback = device.makeBuffer(length: c32Bytes + 128, options: .storageModeShared)
    else { throw Opt0068Error.contract("Metal allocation failed for \(id)") }
    guard a.length >= aBytes, b.length >= bBytes else {
      throw Opt0068Error.contract("Metal input allocation was short")
    }
    self.resultBuffer = c32
    self.readback = readback
    let leftDescriptor = MPSMatrixDescriptor(
      rows: rows, columns: inner, rowBytes: inner * 2, dataType: .float16
    )
    let rightDescriptor = MPSMatrixDescriptor(
      rows: inner, columns: columns, rowBytes: columns * 2, dataType: .float16
    )
    let resultDescriptor = MPSMatrixDescriptor(
      rows: rows, columns: columns, rowBytes: columns * 4, dataType: .float32
    )
    self.left = MPSMatrix(buffer: a, descriptor: leftDescriptor)
    self.right = MPSMatrix(buffer: b, descriptor: rightDescriptor)
    self.result = MPSMatrix(buffer: c32, descriptor: resultDescriptor)
    self.multiplication = MPSMatrixMultiplication(
      device: device,
      transposeLeft: false,
      transposeRight: false,
      resultRows: rows,
      resultColumns: columns,
      interiorColumns: inner,
      alpha: 1,
      beta: 0
    )
  }

  func resetCanaries() {
    memset(readback.contents(), 0xa5, readback.length)
  }

  func output() -> [Float] {
    let count = rows * columns
    return Array(
      UnsafeBufferPointer(
        start: readback.contents().advanced(by: 64).assumingMemoryBound(to: Float.self),
        count: count
      )
    )
  }

  func canaryPassed() -> Bool {
    let bytes = readback.contents().assumingMemoryBound(to: UInt8.self)
    for index in 0..<64 where bytes[index] != 0xa5 { return false }
    for index in (readback.length - 64)..<readback.length where bytes[index] != 0xa5 {
      return false
    }
    return true
  }

  private static func privateBuffer<T>(
    device: MTLDevice, queue: MTLCommandQueue, values: [T]
  ) -> MTLBuffer? {
    let length = values.count * MemoryLayout<T>.stride
    guard let staging = values.withUnsafeBytes({ raw in
      device.makeBuffer(bytes: raw.baseAddress!, length: length, options: .storageModeShared)
    }), let destination = device.makeBuffer(length: length, options: .storageModePrivate),
          let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder()
    else { return nil }
    blit.copy(from: staging, sourceOffset: 0, to: destination, destinationOffset: 0, size: length)
    blit.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    return command.status == .completed ? destination : nil
  }
}

private final class MPSRunner {
  private var ownedDevice: MTLDevice?
  private var ownedQueue: MTLCommandQueue?
  var device: MTLDevice { ownedDevice! }
  var queue: MTLCommandQueue { ownedQueue! }

  init() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
      throw Opt0068Error.contract("Metal device/queue unavailable")
    }
    self.ownedDevice = device
    self.ownedQueue = queue
  }

  func destroy() {
    ownedQueue = nil
    ownedDevice = nil
  }

  func runCombined(
    _ prepared: PreparedCase,
    repetition: Int,
    poisonOutput: Bool = false
  ) throws -> CombinedSample {
    if poisonOutput { try poisonResult(prepared) }
    prepared.resetCanaries()
    guard let command = queue.makeCommandBuffer() else {
      throw Opt0068Error.contract("command buffer unavailable")
    }
    command.label = "OPT-0068 \(prepared.id) complete"
    let startedAtEpochMilliseconds = Int64((Date().timeIntervalSince1970 * 1_000).rounded())
    let encodeStart = ContinuousClock.now
    prepared.multiplication.encode(
      commandBuffer: command,
      leftMatrix: prepared.left,
      rightMatrix: prepared.right,
      resultMatrix: prepared.result
    )
    guard let blit = command.makeBlitCommandEncoder() else {
      throw Opt0068Error.contract("materialization encoder unavailable")
    }
    blit.copy(
      from: prepared.resultBuffer, sourceOffset: 0,
      to: prepared.readback, destinationOffset: 64,
      size: prepared.rows * prepared.columns * 4
    )
    blit.endEncoding()
    let encodingMS = milliseconds(since: encodeStart)
    let wallStart = ContinuousClock.now
    command.commit()
    command.waitUntilCompleted()
    let wallMS = milliseconds(since: wallStart)
    try requireCompleted(command, prepared.id)
    let gpuMS = max(0, (command.gpuEndTime - command.gpuStartTime) * 1_000)
    let output = prepared.output()
    return CombinedSample(
      caseId: prepared.id,
      repetition: repetition,
      startedAtEpochMilliseconds: startedAtEpochMilliseconds,
      completedAtEpochMilliseconds: Int64((Date().timeIntervalSince1970 * 1_000).rounded()),
      flops: prepared.flops,
      encodingMilliseconds: encodingMS,
      commitThroughFenceMilliseconds: wallMS,
      gpuMilliseconds: gpuMS,
      synchronizationOverheadMilliseconds: max(0, wallMS - gpuMS),
      commandThroughFenceMilliseconds: encodingMS + wallMS,
      tflopsIncludingEpilogueAndMaterialization:
        Double(prepared.flops) / max(wallMS, 1e-9) / 1e9,
      outputSha256: sha256(floats: output),
      canaryPassed: prepared.canaryPassed()
    )
  }

  private func poisonResult(_ prepared: PreparedCase) throws {
    guard let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder() else {
      throw Opt0068Error.contract("complete-write poison command unavailable")
    }
    blit.fill(
      buffer: prepared.resultBuffer,
      range: 0..<prepared.resultBuffer.length,
      value: 0xff
    )
    blit.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    try requireCompleted(command, "\(prepared.id) complete-write poison")
  }

  func profileStages(_ prepared: PreparedCase) throws -> StageProfile {
    prepared.resetCanaries()
    let contraction = try timeCommand("\(prepared.id) contraction") { command in
      prepared.multiplication.encode(
        commandBuffer: command,
        leftMatrix: prepared.left,
        rightMatrix: prepared.right,
        resultMatrix: prepared.result
      )
    }
    let epilogue = StageTiming(
      executed: false,
      reason: "ACE repeated-layer dense GEMMs have no bias or activation epilogue",
      encodingMilliseconds: 0,
      commitThroughFenceMilliseconds: 0,
      gpuMilliseconds: 0,
      synchronizationOverheadMilliseconds: 0,
      commandThroughFenceMilliseconds: 0
    )
    let materialization = try timeCommand("\(prepared.id) materialization") { command in
      guard let blit = command.makeBlitCommandEncoder() else {
        throw Opt0068Error.contract("materialization encoder unavailable")
      }
      blit.copy(
        from: prepared.resultBuffer, sourceOffset: 0,
        to: prepared.readback, destinationOffset: 64,
        size: prepared.rows * prepared.columns * 4
      )
      blit.endEncoding()
    }
    return StageProfile(
      caseId: prepared.id,
      contraction: contraction,
      biasEpilogue: epilogue,
      materialization: materialization,
      biasMode: "absent-by-ACE-repeated-layer-contract",
      outputBoundary: "MPS direct FP32 result then explicit guarded materialization"
    )
  }

  private func timeCommand(
    _ label: String, encode: (MTLCommandBuffer) throws -> Void
  ) throws -> StageTiming {
    guard let command = queue.makeCommandBuffer() else {
      throw Opt0068Error.contract("command buffer unavailable")
    }
    command.label = "OPT-0068 \(label)"
    let encodeStart = ContinuousClock.now
    try encode(command)
    let encodingMS = milliseconds(since: encodeStart)
    let wallStart = ContinuousClock.now
    command.commit()
    command.waitUntilCompleted()
    let wallMS = milliseconds(since: wallStart)
    try requireCompleted(command, label)
    return StageTiming(
      executed: true,
      reason: nil,
      encodingMilliseconds: encodingMS,
      commitThroughFenceMilliseconds: wallMS,
      gpuMilliseconds: max(0, (command.gpuEndTime - command.gpuStartTime) * 1_000),
      synchronizationOverheadMilliseconds: max(
        0, wallMS - max(0, (command.gpuEndTime - command.gpuStartTime) * 1_000)
      ),
      commandThroughFenceMilliseconds: encodingMS + wallMS
    )
  }
}

private func prepareActual(
  inputs: AuthenticatedInputs, runner: MPSRunner
) throws -> [PreparedCase] {
  try inputs.cases.map { resolved in
    let activation = try Opt0068Contract.loadActivationFP16(resolved)
    let weight = try Opt0068Contract.unpackWeight(resolved)
    let expected = try Opt0068Contract.loadAcceptedOutput(resolved)
    guard Opt0068Numerics.hasOnlyFiniteValues(expected) else {
      throw Opt0068Error.contract("accepted WebGPU output \(resolved.plan.id) is non-finite")
    }
    return try PreparedCase(
      device: runner.device, queue: runner.queue, resolved: resolved,
      id: resolved.plan.id, rows: Opt0068Frozen.rows,
      inner: resolved.plan.shape.inner, columns: resolved.plan.shape.columns,
      activation: activation, weight: weight, expected: expected
    )
  }
}

private func runCorrectness(
  actual: [PreparedCase], runner: MPSRunner, fullCPU: Bool
) throws -> ([CorrectnessCase], [AdversarialCase], Bool) {
  var cases: [CorrectnessCase] = []
  var allPassed = true
  for prepared in actual {
    let first = try runner.runCombined(prepared, repetition: -1, poisonOutput: true)
    let firstOutput = prepared.output()
    let second = try runner.runCombined(prepared, repetition: -2, poisonOutput: true)
    let secondOutput = prepared.output()
    let versusWebGPU = try Opt0068Numerics.compare(
      candidate: secondOutput, reference: prepared.expected
    )
    var webGPUVersusCPU: NumericMetrics?
    var candidateVersusCPU: NumericMetrics?
    if fullCPU {
      let cpu = acceleratedCPUReference(prepared)
      webGPUVersusCPU = try Opt0068Numerics.compare(
        candidate: prepared.expected, reference: cpu
      )
      candidateVersusCPU = try Opt0068Numerics.compare(candidate: secondOutput, reference: cpu)
    }
    let deterministic = firstOutput.elementsEqual(secondOutput) {
      $0.bitPattern == $1.bitPattern
    }
    let passed = deterministic && first.canaryPassed && second.canaryPassed &&
      versusWebGPU.finiteClassMismatchCount == 0 &&
      versusWebGPU.nonFiniteClassMismatchCount == 0 &&
      versusWebGPU.signedZeroMismatchCount == 0 &&
      versusWebGPU.zeroClassMismatchCount == 0 &&
      (webGPUVersusCPU?.finiteClassMismatchCount ?? 0) == 0 &&
      (webGPUVersusCPU?.nonFiniteClassMismatchCount ?? 0) == 0 &&
      (webGPUVersusCPU?.signedZeroMismatchCount ?? 0) == 0 &&
      (webGPUVersusCPU?.zeroClassMismatchCount ?? 0) == 0 &&
      (candidateVersusCPU?.finiteClassMismatchCount ?? 0) == 0 &&
      (candidateVersusCPU?.nonFiniteClassMismatchCount ?? 0) == 0 &&
      (candidateVersusCPU?.signedZeroMismatchCount ?? 0) == 0 &&
      (candidateVersusCPU?.zeroClassMismatchCount ?? 0) == 0
    allPassed = allPassed && passed
    cases.append(CorrectnessCase(
      id: prepared.id,
      candidateVersusAcceptedWebGPU: versusWebGPU,
      acceptedWebGPUVersusCPU: webGPUVersusCPU,
      candidateVersusCPU: candidateVersusCPU,
      candidateOutputSha256: first.outputSha256,
      repeatedOutputSha256: second.outputSha256,
      deterministic: deterministic,
      canaryPassed: first.canaryPassed && second.canaryPassed,
      passed: passed
    ))
  }

  var adversarial: [AdversarialCase] = []
  for fixture in Opt0068Adversarial.fixtures() {
    let cpu = try Opt0068Numerics.cpuReference(
      activation: fixture.activation, weight: fixture.weight,
      rows: fixture.rows, inner: fixture.inner, columns: fixture.columns
    )
    let prepared = try PreparedCase(
      device: runner.device, queue: runner.queue, resolved: nil,
      id: "adversarial-\(fixture.kind.rawValue)", rows: fixture.rows,
      inner: fixture.inner, columns: fixture.columns,
      activation: fixture.activation, weight: fixture.weight, expected: cpu
    )
    let first = try runner.runCombined(prepared, repetition: -1, poisonOutput: true)
    let firstOutput = prepared.output()
    let second = try runner.runCombined(prepared, repetition: -2, poisonOutput: true)
    let secondOutput = prepared.output()
    let metrics = try Opt0068Numerics.compare(candidate: secondOutput, reference: cpu)
    let deterministic = firstOutput.elementsEqual(secondOutput) {
      $0.bitPattern == $1.bitPattern
    }
    let accumulationMagnitudePassed = fixture.kind != .longKCancellation ||
      metrics.maximumAbsoluteError < 0.01
    let passed = deterministic && first.canaryPassed && second.canaryPassed &&
      metrics.finiteClassMismatchCount == 0 && metrics.nonFiniteClassMismatchCount == 0 &&
      metrics.signedZeroMismatchCount == 0 && metrics.zeroClassMismatchCount == 0 &&
      accumulationMagnitudePassed
    allPassed = allPassed && passed
    adversarial.append(AdversarialCase(
      id: fixture.kind.rawValue, metrics: metrics,
      deterministic: deterministic,
      canaryPassed: first.canaryPassed && second.canaryPassed,
      accumulationMagnitudePassed: accumulationMagnitudePassed,
      passed: passed
    ))
  }
  return (cases, adversarial, allPassed)
}

/// Independent full-output CPU contract. Operands are exact FP16 values widened
/// to FP32; Accelerate performs an FP32 CPU reduction independent of MPS/WebGPU.
/// The smaller adversarial suite separately retains strict source-K-order math.
private func acceleratedCPUReference(_ prepared: PreparedCase) -> [Float] {
  let activation = prepared.activation.map(Float.init)
  let weight = prepared.weight.map(Float.init)
  var output = [Float](repeating: 0, count: prepared.rows * prepared.columns)
  activation.withUnsafeBufferPointer { a in
    weight.withUnsafeBufferPointer { b in
      output.withUnsafeMutableBufferPointer { c in
        vDSP_mmul(
          a.baseAddress!,
          1,
          b.baseAddress!,
          1,
          c.baseAddress!,
          1,
          vDSP_Length(prepared.rows),
          vDSP_Length(prepared.columns),
          vDSP_Length(prepared.inner)
        )
      }
    }
  }
  return output
}

private func runPerformance(
  actual: inout [PreparedCase], runner: MPSRunner, arguments: Arguments
) throws -> (PerformanceSummary, ThermalGateEvidence, ThermalTraceSummary) {
  guard let traceURL = arguments.thermalTrace else {
    throw Opt0068Error.contract("performance modes require --thermal-trace")
  }
  for warmup in 0..<arguments.warmups {
    for prepared in actual { _ = try runner.runCombined(prepared, repetition: -100 - warmup) }
  }
  let monitor = ThermalMonitor(traceURL: traceURL)
  try monitor.start()
  var monitorStopped = false
  defer {
    if !monitorStopped { _ = try? monitor.stop() }
  }
  let gate = try ThermalGateEvidence(
    observations: monitor.awaitNominal(seconds: Opt0068Frozen.gateNominalSeconds)
  )
  var samples: [CombinedSample] = []
  var stageProfiles: [StageProfile] = []
  var intervals: [SustainedInterval]? = nil
  var retention: Double? = nil
  var retentionPassed: Bool? = nil

  if arguments.mode == .measure {
    for repetition in 0..<arguments.samples {
      let order = repetition.isMultiple(of: 2) ? actual : actual.reversed()
      for prepared in order {
        samples.append(try runner.runCombined(prepared, repetition: repetition))
      }
    }
    for prepared in actual { stageProfiles.append(try runner.profileStages(prepared)) }
  } else {
    let start = ContinuousClock.now
    var intervalStart = 0.0
    var intervalFlops: UInt64 = 0
    var allIntervals: [SustainedInterval] = []
    var repetition = 0
    while milliseconds(since: start) < Opt0068Frozen.sustainedSeconds * 1_000 {
      for prepared in actual {
        let sample = try runner.runCombined(prepared, repetition: repetition)
        samples.append(sample)
        intervalFlops += sample.flops
      }
      repetition += 1
      let elapsed = milliseconds(since: start)
      if elapsed - intervalStart >= 5_000 {
        allIntervals.append(SustainedInterval(
          index: allIntervals.count,
          startedMilliseconds: intervalStart,
          completedMilliseconds: elapsed,
          completedFlops: intervalFlops,
          tflops: Double(intervalFlops) / max(elapsed - intervalStart, 1e-9) / 1e9
        ))
        intervalStart = elapsed
        intervalFlops = 0
      }
    }
    intervals = allIntervals
    if allIntervals.count >= 3 {
      let third = max(1, allIntervals.count / 3)
      let first = median(Array(allIntervals.prefix(third)).map(\.tflops))
      let final = median(Array(allIntervals.suffix(third)).map(\.tflops))
      retention = final / max(first, 1e-30)
      retentionPassed = retention! >= Opt0068Frozen.sustainedRetention
    }
  }
  let totalFlops = actual.reduce(UInt64(0)) { $0 + $1.flops }
  // Release every case-owned matrix and Metal buffer while thermal polling remains active.
  actual.removeAll(keepingCapacity: false)
  runner.destroy()
  let cleanupEpochMilliseconds = Int64((Date().timeIntervalSince1970 * 1_000).rounded())
  try monitor.awaitObservation(afterEpochMilliseconds: cleanupEpochMilliseconds)
  let trace = try monitor.stop()
  monitorStopped = true
  let repetitions = Set(samples.map(\.repetition)).filter { $0 >= 0 }.sorted()
  let walls = repetitions.map { repetition in
    samples.filter { $0.repetition == repetition }.reduce(0) {
      $0 + $1.commitThroughFenceMilliseconds
    }
  }
  let rates = walls.map { Double(totalFlops) / max($0, 1e-9) / 1e9 }
  let medianWall = median(walls)
  let medianRate = median(rates)
  return (PerformanceSummary(
    samples: samples,
    stageProfiles: stageProfiles,
    weightedWallMillisecondsByRepetition: walls,
    weightedTFLOPSByRepetition: rates,
    medianWeightedWallMilliseconds: medianWall,
    medianWeightedTFLOPS: medianRate,
    denseRateGatePassed: medianRate >= Opt0068Frozen.requiredDenseTFLOPS,
    webGPUComparisonPerformed: false,
    pairedWebGPUSpeedups: nil,
    denseSpeedupGatePassed: nil,
    completePhaseOneDenseGatePassed: false,
    sustainedIntervals: intervals,
    sustainedRetention: retention,
    sustainedRetentionGatePassed: retentionPassed
  ), gate, trace)
}

private func machineIdentity(
  runner: MPSRunner, harnessCommit: String, harnessSourceRoot: URL
) throws -> MachineIdentity {
  let model = try Opt0068Process.output("/usr/sbin/sysctl", ["-n", "hw.model"])
  let memory = UInt64(try Opt0068Process.output("/usr/sbin/sysctl", ["-n", "hw.memsize"])) ?? 0
  let osVersion = try Opt0068Process.output("/usr/bin/sw_vers", ["-productVersion"])
  let osBuild = try Opt0068Process.output("/usr/bin/sw_vers", ["-buildVersion"])
  guard model == "Mac15,12", memory == 17_179_869_184,
        osVersion == "26.5.2", osBuild == "25F84" else {
    throw Opt0068Error.contract(
      "OPT-0068 requires Mac15,12 / 17179869184 bytes / macOS 26.5.2 (25F84); got " +
      "\(model) / \(memory) / \(osVersion) (\(osBuild))"
    )
  }
  let displayJSON = try Opt0068Process.output(
    "/usr/sbin/system_profiler", ["SPDisplaysDataType", "-json"]
  )
  let displayObject = try JSONSerialization.jsonObject(with: Data(displayJSON.utf8))
  let displays = (displayObject as? [String: Any])?["SPDisplaysDataType"] as? [[String: Any]]
  let gpuCoreCount = displays?.compactMap { display -> Int? in
    let raw = display["sppci_cores"] as? String
    return raw.flatMap { Int($0.filter(\.isNumber)) }
  }.first ?? 0
  guard gpuCoreCount == 10 else {
    throw Opt0068Error.contract("OPT-0068 requires 10 GPU cores; system_profiler reported \(gpuCoreCount)")
  }
  let familyChecks: [(String, MTLGPUFamily)] = [
    ("apple1", .apple1), ("apple2", .apple2), ("apple3", .apple3),
    ("apple4", .apple4), ("apple5", .apple5), ("apple6", .apple6),
    ("apple7", .apple7), ("apple8", .apple8), ("apple9", .apple9),
    ("mac2", .mac2), ("common1", .common1),
    ("common2", .common2), ("common3", .common3), ("metal3", .metal3),
  ]
  let metalFamilies = familyChecks.filter { runner.device.supportsFamily($0.1) }.map(\.0)
  let mpsInfo = Bundle(
    path: "/System/Library/Frameworks/MetalPerformanceShaders.framework"
  )?.infoDictionary ?? [:]
  let sourcePaths = [
    "Package.swift",
    "Sources/AceOpt0068Core/Contract.swift",
    "Sources/AceOpt0068Core/Numerics.swift",
    "Sources/AceOpt0068Core/Thermal.swift",
    "Sources/AceOpt0068MPS/main.swift",
  ]
  let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
  return MachineIdentity(
    model: model, memoryBytes: memory, osVersion: osVersion, osBuild: osBuild,
    metalDevice: runner.device.name,
    gpuCoreCount: gpuCoreCount,
    metalFamilies: metalFamilies,
    mpsFrameworkVersion: mpsInfo["CFBundleShortVersionString"] as? String ?? "unknown",
    mpsFrameworkSDK: mpsInfo["DTSDKName"] as? String ?? "unknown",
    swiftVersion: try Opt0068Process.output("/usr/bin/xcrun", ["swift", "--version"]),
    sdkPath: try Opt0068Process.output("/usr/bin/xcrun", ["--show-sdk-path"]),
    executableSha256: try Opt0068Contract.sha256(file: executable),
    sourceBundleSha256: try Opt0068Contract.sourceBundleSHA256(
      root: harnessSourceRoot, relativePaths: sourcePaths
    ),
    harnessCommit: harnessCommit
  )
}

private func describe() {
  print("""
  OPT-0068 static Phase-1 native dense gate.

  No actual M2250 activation fixture is committed. GPU modes fail closed until the
  capture recipe's six FP32 activation boundaries and nine accepted WebGPU FP32
  outputs are present and authenticated by an ace-opt-0068-m2250-native-fixture-v1
  manifest. This executable never invents zero or synthetic throughput operands.

  Frozen exact family: M2250, K/N multiplicities 2048/2048 x4, 2048/1024 x2,
  2048/6144 x2, 6144/2048 x1. Repeated-layer bias is absent. MPS contracts FP16
  inputs with its opaque reduction directly into an FP32 result matrix. Bias and
  activation epilogue are explicitly reported as absent for these repeated-layer
  GEMMs; a guarded blit is the separately timed materialization. MPS mixed-input/
  result support and FP32-accumulation behavior are proven only by executing the
  mandatory screens; the harness fails closed if either premise is unsupported.
  """)
}

private func main() throws {
  let arguments = try Arguments.parse()
  if arguments.mode == .describe { describe(); return }
  guard let packageDirectory = arguments.packageDirectory,
        let fixtureManifest = arguments.fixtureManifest else {
    throw Opt0068Error.contract("non-describe modes require --package-dir and --fixture-manifest")
  }
  let inputs = try Opt0068Contract.authenticate(
    packageDirectory: packageDirectory, fixtureManifest: fixtureManifest
  )
  if arguments.mode == .inspect {
    print("authenticated \(inputs.cases.count) actual M2250 cases; no Metal device was created")
    return
  }
  guard arguments.consent == Opt0068Frozen.executionConsent else {
    throw Opt0068Error.contract(
      "native GPU execution requires --execute-native-gpu \(Opt0068Frozen.executionConsent)"
    )
  }
  guard let output = arguments.output, let commit = arguments.harnessCommit,
        let harnessSourceRoot = arguments.harnessSourceRoot,
        commit.count == 40, arguments.samples > 0, arguments.warmups >= 0 else {
    throw Opt0068Error.contract(
      "GPU modes require output, 40-hex commit, harness source root, positive samples, and warmups"
    )
  }
  guard commit.allSatisfy(\.isHexDigit) else {
    throw Opt0068Error.contract("harness commit must be lowercase/uppercase hexadecimal")
  }
  guard !FileManager.default.fileExists(atPath: output.path) else {
    throw Opt0068Error.contract("receipt already exists: \(output.path)")
  }
  if arguments.mode != .correctness && !arguments.fullCPUContract {
    throw Opt0068Error.contract("performance modes require --full-cpu-contract before timing")
  }
  let runner = try MPSRunner()
  let identity = try machineIdentity(
    runner: runner, harnessCommit: commit, harnessSourceRoot: harnessSourceRoot
  )
  var actual = try prepareActual(inputs: inputs, runner: runner)
  let correctness = try runCorrectness(
    actual: actual, runner: runner, fullCPU: arguments.fullCPUContract
  )
  guard correctness.2 else {
    throw Opt0068Error.contract("mandatory numerical/class/determinism/canary screen failed; timing refused")
  }
  var performance: PerformanceSummary?
  var thermalGate: ThermalGateEvidence?
  var thermal: ThermalTraceSummary?
  if arguments.mode == .measure || arguments.mode == .sustained {
    (performance, thermalGate, thermal) = try runPerformance(
      actual: &actual, runner: runner, arguments: arguments
    )
  } else {
    actual.removeAll(keepingCapacity: false)
    runner.destroy()
  }
  let receipt = Receipt(
    schema: Opt0068Frozen.receiptSchema,
    experimentId: Opt0068Frozen.experimentID,
    status: "benchmark-only",
    disposition: "no native prototype, browser selection, or product claim",
    claimBoundary: [
      "dense primitive only",
      "MPS opaque reduction with FP16 inputs and requested FP32 result",
      "no complete evaluation, VAE, waveform, listening, or Generate-to-WAV evidence",
      "a dense pass authorizes only the complete native prototype declared by OPT-0068",
    ],
    identity: identity,
    packageManifestSha256: Opt0068Frozen.packageManifestSHA256,
    activationFixtureManifestSha256: try Opt0068Contract.sha256(file: fixtureManifest),
    arithmetic: [
      "activation": "captured production FP32 rounded once to IEEE FP16 before timing",
      "weight": "authenticated revision-7 FP16, repacked N256/K32 to row-major before timing",
      "accumulation": "MPS opaque; admissibility requires FP32-accumulation adversarial evidence",
      "independentCPU": "Accelerate FP32 SGEMM over exact FP16-widened actual operands; scalar source-K-order adversarial oracle",
      "result": "MPS direct FP32 result matrix; runtime support must pass all screens",
      "bias": "absent-by-ACE-repeated-layer-contract",
      "materialization": "private FP32 buffer to shared guarded readback",
    ],
    correctness: correctness.0,
    adversarial: correctness.1,
    performance: performance,
    thermalGate: thermalGate,
    thermal: thermal,
    fullCPUContractExecuted: arguments.fullCPUContract,
    allMandatoryScreensPassed: correctness.2 && arguments.fullCPUContract
  )
  try FileManager.default.createDirectory(
    at: output.deletingLastPathComponent(), withIntermediateDirectories: true
  )
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
  var data = try encoder.encode(receipt)
  data.append(0x0a)
  try data.write(to: output, options: .withoutOverwriting)
  print("wrote \(output.path)")
}

private func milliseconds(since start: ContinuousClock.Instant) -> Double {
  let duration = start.duration(to: .now)
  return Double(duration.components.seconds) * 1_000 +
    Double(duration.components.attoseconds) / 1e15
}

private func median(_ values: [Double]) -> Double {
  guard !values.isEmpty else { return 0 }
  let sorted = values.sorted()
  let middle = sorted.count / 2
  return sorted.count.isMultiple(of: 2)
    ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
}

private func sha256(floats: [Float]) -> String {
  var hasher = SHA256()
  for value in floats {
    var bits = value.bitPattern.littleEndian
    withUnsafeBytes(of: &bits) { hasher.update(bufferPointer: $0) }
  }
  return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}

private func requireCompleted(_ command: MTLCommandBuffer, _ label: String) throws {
  guard command.status == .completed, command.error == nil else {
    throw Opt0068Error.contract("\(label) command failed: \(String(describing: command.error))")
  }
}

do {
  try main()
} catch {
  FileHandle.standardError.write(Data("OPT-0068 FAIL CLOSED: \(error)\n".utf8))
  exit(2)
}
