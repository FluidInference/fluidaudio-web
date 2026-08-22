import CryptoKit
import Foundation

public enum Opt0068Error: Error, CustomStringConvertible, Sendable {
  case contract(String)

  public var description: String {
    switch self {
    case .contract(let message): return message
    }
  }
}

public enum Opt0068Frozen {
  public static let experimentID = "OPT-0068"
  public static let fixtureSchema = "ace-opt-0068-m2250-native-fixture-v1"
  public static let receiptSchema = "ace-opt-0068-native-dense-receipt-v1"
  public static let packageFormat = "ace-step-webgpu-v1"
  public static let packageProfile = "fp16-dit-dense-experimental"
  public static let packageManifestSHA256 =
    "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f"
  public static let packageConverterRevision = 7
  public static let aceSourceCommit = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0"
  public static let aceSnapshot = "19671f406d603126926c1b7e2adc169acbcade22"
  public static let mainManifestSHA256 =
    "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6"
  public static let fixtureManifestSHA256 =
    "cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb"
  public static let requestSHA256 =
    "031e418ac5db37355fe5e265a005cb280e02ce418e560312ac89fa184bb8862f"
  public static let requestByteLength = 366
  public static let evaluation0SHA256 =
    "d7f4280fdc43a038728df167f02819c35d99dac812347731d2fb8ac421a36286"
  public static let evaluation0ElementCount = 288_000
  public static let rows = 2_250
  public static let conditionTokens = 98
  public static let layer = 0
  public static let evaluation = 0
  public static let gateNominalSeconds = 30.0
  public static let thermalPollMilliseconds = 900
  public static let maximumThermalPollGapMilliseconds: Int64 = 1_800
  public static let sustainedSeconds = 60.0
  public static let sustainedRetention = 0.80
  public static let requiredDenseWallSpeedup = 1.40
  public static let requiredDenseTFLOPS = 2.40
  public static let executionConsent = "I_UNDERSTAND_OPT_0068_BENCHMARK_ONLY"
}

public struct DenseShape: Codable, Hashable, Sendable {
  public let inner: Int
  public let columns: Int

  public init(inner: Int, columns: Int) {
    self.inner = inner
    self.columns = columns
  }

  public var key: String { "K\(inner)-N\(columns)" }
  public var flops: UInt64 {
    UInt64(2) * UInt64(Opt0068Frozen.rows) * UInt64(inner) * UInt64(columns)
  }
}

public enum Opt0068DensePlan {
  public static let shapes: [DenseShape: Int] = [
    DenseShape(inner: 2_048, columns: 2_048): 4,
    DenseShape(inner: 2_048, columns: 1_024): 2,
    DenseShape(inner: 2_048, columns: 6_144): 2,
    DenseShape(inner: 6_144, columns: 2_048): 1,
  ]

  public static let cases: [FrozenCase] = [
    FrozenCase(
      id: "self-query", activationID: "self-modulated",
      outputID: "self-query", tensor: "ace.decoder.layers.0.self_attn.q_proj.weight",
      shape: DenseShape(inner: 2_048, columns: 2_048)
    ),
    FrozenCase(
      id: "self-key", activationID: "self-modulated",
      outputID: "self-key", tensor: "ace.decoder.layers.0.self_attn.k_proj.weight",
      shape: DenseShape(inner: 2_048, columns: 1_024)
    ),
    FrozenCase(
      id: "self-value", activationID: "self-modulated",
      outputID: "self-value", tensor: "ace.decoder.layers.0.self_attn.v_proj.weight",
      shape: DenseShape(inner: 2_048, columns: 1_024)
    ),
    FrozenCase(
      id: "self-output", activationID: "self-merged-attention",
      outputID: "self-output", tensor: "ace.decoder.layers.0.self_attn.o_proj.weight",
      shape: DenseShape(inner: 2_048, columns: 2_048)
    ),
    FrozenCase(
      id: "cross-query", activationID: "cross-normalized",
      outputID: "cross-query", tensor: "ace.decoder.layers.0.cross_attn.q_proj.weight",
      shape: DenseShape(inner: 2_048, columns: 2_048)
    ),
    FrozenCase(
      id: "cross-output", activationID: "cross-merged-attention",
      outputID: "cross-output", tensor: "ace.decoder.layers.0.cross_attn.o_proj.weight",
      shape: DenseShape(inner: 2_048, columns: 2_048)
    ),
    FrozenCase(
      id: "mlp-gate", activationID: "mlp-modulated",
      outputID: "mlp-gate", tensor: "ace.decoder.layers.0.mlp.gate_proj.weight",
      shape: DenseShape(inner: 2_048, columns: 6_144)
    ),
    FrozenCase(
      id: "mlp-up", activationID: "mlp-modulated",
      outputID: "mlp-up", tensor: "ace.decoder.layers.0.mlp.up_proj.weight",
      shape: DenseShape(inner: 2_048, columns: 6_144)
    ),
    FrozenCase(
      id: "mlp-down", activationID: "mlp-gated-activation",
      outputID: "mlp-down", tensor: "ace.decoder.layers.0.mlp.down_proj.weight",
      shape: DenseShape(inner: 6_144, columns: 2_048)
    ),
  ]

  public struct FrozenCase: Hashable, Sendable {
    public let id: String
    public let activationID: String
    public let outputID: String
    public let tensor: String
    public let shape: DenseShape
  }
}

public struct HashedFile: Codable, Hashable, Sendable {
  public let id: String
  public let path: String
  public let dtype: String
  public let shape: [Int]
  public let elementCount: Int
  public let byteLength: Int
  public let sha256: String
  public let finiteCount: Int
  public let nonzeroCount: Int
  public let minimum: Double
  public let maximum: Double
  public let headF32Bits: [String]
  public let tailF32Bits: [String]
}

public struct FixtureCase: Codable, Hashable, Sendable {
  public let id: String
  public let activation: String
  public let acceptedWebGPUOutput: String
  public let weightTensor: String
  public let rows: Int
  public let inner: Int
  public let columns: Int
}

public struct FixtureAuthority: Codable, Hashable, Sendable {
  public let experimentId: String
  public let aceSourceCommit: String
  public let aceSnapshot: String
  public let mainManifestSha256: String
  public let goldenFixtureManifestSha256: String
  public let packageManifestSha256: String
  public let packageConverterRevision: Int
  public let requestId: String
  public let requestSha256: String
  public let requestByteLength: Int
  public let plannerEnabled: Bool
  public let durationSeconds: Int
  public let sampler: String
  public let dcwMode: String
  public let lowFrequencyStrength: Double
  public let highFrequencyStrength: Double
  public let evaluation: Int
  public let layer: Int
  public let conditionTokens: Int
  public let expectedEvaluation0Sha256: String
  public let captureCommit: String
  public let captureSourceSha256: String
}

public struct ActivationFixture: Codable, Sendable {
  public let schema: String
  public let authority: FixtureAuthority
  public let evaluationOutput: HashedFile
  public let activations: [HashedFile]
  public let acceptedWebGPUOutputs: [HashedFile]
  public let cases: [FixtureCase]
}

public struct PackageManifest: Decodable, Sendable {
  public struct Provenance: Decodable, Sendable {
    public let aceSnapshot: String
    public let converterRevision: Int
    public let referenceCommit: String
  }

  public struct FileEntry: Decodable, Sendable {
    public let byteLength: Int
    public let kind: String
    public let name: String
    public let sha256: String
  }

  public struct TensorEntry: Decodable, Sendable {
    public let byteLength: Int
    public let byteOffset: Int
    public let dtype: String
    public let layout: String
    public let lifetime: String
    public let logicalShape: [Int]
    public let logicalTensor: String
    public let phase: String
    public let shard: String
    public let transformation: String
  }

  public let format: String
  public let profile: String
  public let provenance: Provenance
  public let files: [FileEntry]
  public let tensors: [String: TensorEntry]
}

public struct ResolvedCase: Sendable {
  public let plan: Opt0068DensePlan.FrozenCase
  public let activation: URL
  public let acceptedOutput: URL
  public let tensor: PackageManifest.TensorEntry
  public let shard: URL
}

public struct AuthenticatedInputs: Sendable {
  public let fixture: ActivationFixture
  public let package: PackageManifest
  public let cases: [ResolvedCase]
}

public enum Opt0068Contract {
  public static func authenticate(
    packageDirectory: URL,
    fixtureManifest: URL
  ) throws -> AuthenticatedInputs {
    let packageManifestURL = packageDirectory.appendingPathComponent("manifest.json")
    try requireRegularFile(packageManifestURL, "revision-7 package manifest")
    let manifestHash = try sha256(file: packageManifestURL)
    try require(
      manifestHash == Opt0068Frozen.packageManifestSHA256,
      "package manifest SHA-256 \(manifestHash) is not frozen revision 7"
    )
    let decoder = JSONDecoder()
    let package = try decoder.decode(
      PackageManifest.self,
      from: Data(contentsOf: packageManifestURL, options: .mappedIfSafe)
    )
    try validate(package: package)

    try requireRegularFile(fixtureManifest, "actual M2250 activation fixture manifest")
    let fixture = try decoder.decode(
      ActivationFixture.self,
      from: Data(contentsOf: fixtureManifest, options: .mappedIfSafe)
    )
    try validate(fixture: fixture)
    let fixtureRoot = fixtureManifest.deletingLastPathComponent()
    let activations = Dictionary(uniqueKeysWithValues: fixture.activations.map { ($0.id, $0) })
    let outputs = Dictionary(
      uniqueKeysWithValues: fixture.acceptedWebGPUOutputs.map { ($0.id, $0) }
    )
    var verifiedFiles = Set<String>()
    var verifiedShards = Set<String>()
    var resolved: [ResolvedCase] = []

    try validate(
      file: fixture.evaluationOutput,
      expectedID: "evaluation-0-result",
      expectedPath: "evaluation-0-result.f32le",
      expectedShape: [Opt0068Frozen.evaluation0ElementCount],
      root: fixtureRoot,
      verified: &verifiedFiles
    )
    try require(
      fixture.evaluationOutput.sha256 == Opt0068Frozen.evaluation0SHA256,
      "capture final evaluation-0 result does not match the frozen OPT-0067 identity"
    )

    for frozen in Opt0068DensePlan.cases {
      guard let declared = fixture.cases.first(where: { $0.id == frozen.id }) else {
        throw Opt0068Error.contract("fixture omits exact case \(frozen.id)")
      }
      try require(
        declared.activation == frozen.activationID &&
          declared.acceptedWebGPUOutput == frozen.outputID &&
          declared.weightTensor == frozen.tensor &&
          declared.rows == Opt0068Frozen.rows &&
          declared.inner == frozen.shape.inner &&
          declared.columns == frozen.shape.columns,
        "fixture case \(frozen.id) changed the frozen operation"
      )
      guard let activation = activations[frozen.activationID] else {
        throw Opt0068Error.contract("fixture omits activation \(frozen.activationID)")
      }
      guard let output = outputs[frozen.outputID] else {
        throw Opt0068Error.contract("fixture omits accepted output \(frozen.outputID)")
      }
      try validate(
        file: activation,
        expectedID: frozen.activationID,
        expectedPath: "activation-\(frozen.activationID).f32le",
        expectedShape: [Opt0068Frozen.rows, frozen.shape.inner],
        root: fixtureRoot,
        verified: &verifiedFiles
      )
      try validate(
        file: output,
        expectedID: frozen.outputID,
        expectedPath: "output-\(frozen.outputID).f32le",
        expectedShape: [Opt0068Frozen.rows, frozen.shape.columns],
        root: fixtureRoot,
        verified: &verifiedFiles
      )
      guard let tensor = package.tensors[frozen.tensor] else {
        throw Opt0068Error.contract("package omits \(frozen.tensor)")
      }
      try validate(tensor: tensor, frozen: frozen)
      guard let file = package.files.first(where: { $0.name == tensor.shard }) else {
        throw Opt0068Error.contract("manifest omits shard entry \(tensor.shard)")
      }
      let shard = packageDirectory.appendingPathComponent(tensor.shard)
      if verifiedShards.insert(tensor.shard).inserted {
        try requireRegularFile(shard, "package shard \(tensor.shard)")
        let attributes = try FileManager.default.attributesOfItem(atPath: shard.path)
        let length = (attributes[.size] as? NSNumber)?.intValue
        try require(length == file.byteLength, "shard \(tensor.shard) byte length changed")
        let hash = try sha256(file: shard)
        try require(hash == file.sha256, "shard \(tensor.shard) SHA-256 changed")
      }
      resolved.append(ResolvedCase(
        plan: frozen,
        activation: fixtureRoot.appendingPathComponent(activation.path),
        acceptedOutput: fixtureRoot.appendingPathComponent(output.path),
        tensor: tensor,
        shard: shard
      ))
    }
    return AuthenticatedInputs(fixture: fixture, package: package, cases: resolved)
  }

  public static func unpackWeight(_ resolved: ResolvedCase) throws -> [Float16] {
    let entry = resolved.tensor
    let data = try readRegion(
      file: resolved.shard,
      offset: entry.byteOffset,
      length: entry.byteLength
    )
    let shape = resolved.plan.shape
    let scalarCount = shape.inner * shape.columns
    try require(data.count == scalarCount * 2, "weight byte count changed")
    let physical: [UInt16] = data.withUnsafeBytes { raw in
      Array(raw.bindMemory(to: UInt16.self))
    }
    var rowMajorB = [Float16](repeating: 0, count: scalarCount)
    let innerTiles = shape.inner / 32
    for column in 0..<shape.columns {
      let columnTile = column / 256
      let columnInTile = column % 256
      for inner in 0..<shape.inner {
        let physicalIndex =
          (((columnTile * innerTiles + inner / 32) * 32 + inner % 32) * 256) +
          columnInTile
        rowMajorB[inner * shape.columns + column] =
          Float16(bitPattern: UInt16(littleEndian: physical[physicalIndex]))
      }
    }
    return rowMajorB
  }

  public static func loadActivationFP16(_ resolved: ResolvedCase) throws -> [Float16] {
    let data = try Data(contentsOf: resolved.activation, options: .mappedIfSafe)
    return data.withUnsafeBytes { raw in
      raw.bindMemory(to: UInt32.self).map {
        Float16(Float(bitPattern: UInt32(littleEndian: $0)))
      }
    }
  }

  public static func loadAcceptedOutput(_ resolved: ResolvedCase) throws -> [Float] {
    let data = try Data(contentsOf: resolved.acceptedOutput, options: .mappedIfSafe)
    return data.withUnsafeBytes { raw in
      raw.bindMemory(to: UInt32.self).map {
        Float(bitPattern: UInt32(littleEndian: $0))
      }
    }
  }

  public static func physicalWeightIndex(
    column: Int,
    inner: Int,
    shape: DenseShape
  ) -> Int {
    (((column / 256) * (shape.inner / 32) + inner / 32) * 32 + inner % 32) * 256 +
      column % 256
  }

  public static func sha256(file: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: file)
    defer { try? handle.close() }
    var hasher = SHA256()
    while let data = try handle.read(upToCount: 4 * 1_024 * 1_024), !data.isEmpty {
      hasher.update(data: data)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }

  public static func sourceBundleSHA256(root: URL, relativePaths: [String]) throws -> String {
    var hasher = SHA256()
    for path in relativePaths.sorted() {
      let file = root.appendingPathComponent(path)
      try requireRegularFile(file, "harness source \(path)")
      let pathData = Data(path.utf8)
      var pathLength = UInt64(pathData.count).littleEndian
      withUnsafeBytes(of: &pathLength) { hasher.update(bufferPointer: $0) }
      hasher.update(data: pathData)
      let data = try Data(contentsOf: file, options: .mappedIfSafe)
      var byteLength = UInt64(data.count).littleEndian
      withUnsafeBytes(of: &byteLength) { hasher.update(bufferPointer: $0) }
      hasher.update(data: data)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }

  private static func validate(package: PackageManifest) throws {
    try require(package.format == Opt0068Frozen.packageFormat, "package format changed")
    try require(package.profile == Opt0068Frozen.packageProfile, "package profile changed")
    try require(
      package.provenance.converterRevision == Opt0068Frozen.packageConverterRevision,
      "package converter revision changed"
    )
    try require(
      package.provenance.referenceCommit == Opt0068Frozen.aceSourceCommit,
      "ACE source commit changed"
    )
    try require(
      package.provenance.aceSnapshot == Opt0068Frozen.aceSnapshot,
      "ACE model snapshot changed"
    )
  }

  private static func validate(fixture: ActivationFixture) throws {
    try require(fixture.schema == Opt0068Frozen.fixtureSchema, "fixture schema changed")
    let a = fixture.authority
    try require(a.experimentId == Opt0068Frozen.experimentID, "fixture experiment changed")
    try require(a.aceSourceCommit == Opt0068Frozen.aceSourceCommit, "fixture ACE source changed")
    try require(a.aceSnapshot == Opt0068Frozen.aceSnapshot, "fixture ACE snapshot changed")
    try require(
      a.mainManifestSha256 == Opt0068Frozen.mainManifestSHA256,
      "fixture main package authority changed"
    )
    try require(
      a.goldenFixtureManifestSha256 == Opt0068Frozen.fixtureManifestSHA256,
      "fixture golden authority changed"
    )
    try require(
      a.packageManifestSha256 == Opt0068Frozen.packageManifestSHA256,
      "fixture package authority changed"
    )
    try require(
      a.packageConverterRevision == Opt0068Frozen.packageConverterRevision,
      "fixture converter revision changed"
    )
    try require(
      a.requestId == "ace-turbo-v1-correctness" &&
        a.requestSha256 == Opt0068Frozen.requestSHA256 &&
        a.requestByteLength == Opt0068Frozen.requestByteLength && !a.plannerEnabled &&
        a.durationSeconds == 180 && a.sampler == "shift-3-euler-8-evaluations" &&
        a.dcwMode == "double-haar" && a.lowFrequencyStrength == 0.05 &&
        a.highFrequencyStrength == 0.02,
      "fixture request is not the frozen 180-second direct authority"
    )
    try require(
      a.evaluation == Opt0068Frozen.evaluation && a.layer == Opt0068Frozen.layer &&
        a.conditionTokens == Opt0068Frozen.conditionTokens,
      "fixture graph boundary changed"
    )
    try require(
      a.expectedEvaluation0Sha256 == Opt0068Frozen.evaluation0SHA256,
      "fixture final evaluation-0 authority changed"
    )
    try require(
      isLowercaseHex(a.captureCommit, count: 40),
      "fixture capture commit must be 40 lowercase hex characters"
    )
    try require(
      isLowercaseHex(a.captureSourceSha256, count: 64),
      "fixture capture source hash must be 64 lowercase hex characters"
    )
    try require(
      Set(fixture.activations.map(\.id)).count == fixture.activations.count &&
        Set(fixture.activations.map(\.path)).count == fixture.activations.count,
      "duplicate activation fixture id/path"
    )
    try require(
      Set(fixture.acceptedWebGPUOutputs.map(\.id)).count ==
        fixture.acceptedWebGPUOutputs.count &&
        Set(fixture.acceptedWebGPUOutputs.map(\.path)).count ==
        fixture.acceptedWebGPUOutputs.count,
      "duplicate accepted-output fixture id/path"
    )
    try require(
      Set(fixture.activations.map(\.id)) == Set(Opt0068DensePlan.cases.map(\.activationID)),
      "activation fixture IDs changed"
    )
    try require(
      Set(fixture.acceptedWebGPUOutputs.map(\.id)) ==
        Set(Opt0068DensePlan.cases.map(\.outputID)),
      "accepted-output fixture IDs changed"
    )
    let everyPath = fixture.activations.map(\.path) +
      fixture.acceptedWebGPUOutputs.map(\.path) + [fixture.evaluationOutput.path]
    try require(Set(everyPath).count == 16, "captured activation/output paths must be distinct")
    try require(Set(fixture.cases.map(\.id)).count == fixture.cases.count, "duplicate fixture case")
    try require(fixture.cases.count == Opt0068DensePlan.cases.count, "fixture case count changed")
    let observed = Dictionary(grouping: fixture.cases) {
      DenseShape(inner: $0.inner, columns: $0.columns)
    }.mapValues(\.count)
    try require(observed == Opt0068DensePlan.shapes, "fixture lost the 4/2/2/1 shape mix")
  }

  private static func validate(
    file: HashedFile,
    expectedID: String,
    expectedPath: String,
    expectedShape: [Int],
    root: URL,
    verified: inout Set<String>
  ) throws {
    try require(file.id == expectedID, "captured file ID changed from \(expectedID)")
    try require(file.path == expectedPath, "\(file.id) capture filename changed")
    try require(file.dtype == "float32-le", "\(file.id) must be captured FP32 LE")
    try require(file.shape == expectedShape, "\(file.id) shape changed")
    let elements = expectedShape.reduce(1, *)
    try require(file.elementCount == elements, "\(file.id) element count changed")
    try require(file.byteLength == elements * 4, "\(file.id) byte length changed")
    try require(file.path == (file.path as NSString).lastPathComponent, "\(file.id) path escapes fixture root")
    let url = root.appendingPathComponent(file.path)
    if verified.insert(file.path).inserted {
      try requireRegularFile(url, "fixture file \(file.path)")
      let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
      let length = (attributes[.size] as? NSNumber)?.intValue
      try require(length == file.byteLength, "\(file.id) file byte length changed")
      let hash = try sha256(file: url)
      try require(hash == file.sha256, "\(file.id) SHA-256 changed")
      try validateFloatEvidence(file: file, url: url, elements: elements)
    }
  }

  private static func validateFloatEvidence(
    file: HashedFile,
    url: URL,
    elements: Int
  ) throws {
    try require(
      file.finiteCount == elements && file.nonzeroCount > 0 &&
        file.nonzeroCount <= elements,
      "\(file.id) is non-finite, all-zero, or has invalid evidence counts"
    )
    try require(
      file.minimum.isFinite && file.maximum.isFinite && file.minimum <= file.maximum,
      "\(file.id) range evidence is invalid"
    )
    try require(
      file.headF32Bits.count == 8 && file.tailF32Bits.count == 8 &&
        file.headF32Bits.allSatisfy { isLowercaseHex($0, count: 8) } &&
        file.tailF32Bits.allSatisfy { isLowercaseHex($0, count: 8) },
      "\(file.id) bounded diagnostic slices are invalid"
    )
    let data = try Data(contentsOf: url, options: .mappedIfSafe)
    var finiteCount = 0
    var nonzeroCount = 0
    var minimum = Double.infinity
    var maximum = -Double.infinity
    var head: [String] = []
    var tail: [String] = []
    data.withUnsafeBytes { raw in
      for index in 0..<elements {
        let bits = UInt32(littleEndian: raw.loadUnaligned(
          fromByteOffset: index * 4, as: UInt32.self
        ))
        let value = Float(bitPattern: bits)
        if value.isFinite { finiteCount += 1 }
        if value != 0 { nonzeroCount += 1 }
        minimum = Swift.min(minimum, Double(value))
        maximum = Swift.max(maximum, Double(value))
        if index < 8 { head.append(String(format: "%08x", bits)) }
        if index >= elements - 8 { tail.append(String(format: "%08x", bits)) }
      }
    }
    try require(
      finiteCount == file.finiteCount && nonzeroCount == file.nonzeroCount &&
        minimum == file.minimum && maximum == file.maximum &&
        head == file.headF32Bits && tail == file.tailF32Bits,
      "\(file.id) float evidence does not match authenticated bytes"
    )
  }

  private static func validate(
    tensor: PackageManifest.TensorEntry,
    frozen: Opt0068DensePlan.FrozenCase
  ) throws {
    try require(tensor.logicalTensor == frozen.tensor, "tensor logical name changed")
    try require(
      tensor.logicalShape == [frozen.shape.columns, frozen.shape.inner],
      "tensor \(frozen.tensor) logical shape changed"
    )
    try require(tensor.dtype == "float16", "tensor \(frozen.tensor) is not FP16")
    try require(
      tensor.layout == "dit-gemm-n256-k32-tile-major-v1",
      "tensor \(frozen.tensor) layout changed"
    )
    try require(
      tensor.transformation == "bf16-to-ieee-fp16-dit-gemm-n256-k32-tile-major-v1",
      "tensor \(frozen.tensor) transformation changed"
    )
    try require(tensor.phase == "dit" && tensor.lifetime == "dit", "tensor phase changed")
    try require(
      tensor.byteLength == frozen.shape.inner * frozen.shape.columns * 2,
      "tensor \(frozen.tensor) byte length changed"
    )
  }

  private static func readRegion(file: URL, offset: Int, length: Int) throws -> Data {
    let handle = try FileHandle(forReadingFrom: file)
    defer { try? handle.close() }
    try handle.seek(toOffset: UInt64(offset))
    let data = try handle.read(upToCount: length) ?? Data()
    try require(data.count == length, "short read from \(file.lastPathComponent)")
    return data
  }

  private static func requireRegularFile(_ url: URL, _ label: String) throws {
    var isDirectory: ObjCBool = false
    try require(
      FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) &&
        !isDirectory.boolValue,
      "missing \(label) at \(url.path)"
    )
  }

  private static func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw Opt0068Error.contract(message) }
  }

  private static func isLowercaseHex(_ value: String, count: Int) -> Bool {
    value.count == count && value.allSatisfy { ("0"..."9").contains($0) || ("a"..."f").contains($0) }
  }
}
