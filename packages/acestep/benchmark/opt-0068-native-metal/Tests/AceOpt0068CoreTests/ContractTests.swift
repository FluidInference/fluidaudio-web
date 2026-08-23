import Foundation
import Testing
@testable import AceOpt0068Core

@Test func frozenShapeMixIsExactlyFourTwoTwoOne() {
  #expect(Opt0068DensePlan.cases.count == 9)
  let observed = Dictionary(grouping: Opt0068DensePlan.cases, by: \.shape).mapValues(\.count)
  #expect(observed == Opt0068DensePlan.shapes)
  #expect(observed[DenseShape(inner: 2_048, columns: 2_048)] == 4)
  #expect(observed[DenseShape(inner: 2_048, columns: 1_024)] == 2)
  #expect(observed[DenseShape(inner: 2_048, columns: 6_144)] == 2)
  #expect(observed[DenseShape(inner: 6_144, columns: 2_048)] == 1)
}

@Test func n256K32MappingIsBijectiveAcrossTileBoundaries() {
  let shape = DenseShape(inner: 64, columns: 512)
  var seen = Set<Int>()
  for column in 0..<shape.columns {
    for inner in 0..<shape.inner {
      seen.insert(Opt0068Contract.physicalWeightIndex(
        column: column, inner: inner, shape: shape
      ))
    }
  }
  #expect(seen.count == shape.inner * shape.columns)
  #expect(seen.min() == 0)
  #expect(seen.max() == shape.inner * shape.columns - 1)
  #expect(Opt0068Contract.physicalWeightIndex(column: 0, inner: 31, shape: shape) == 31 * 256)
  #expect(Opt0068Contract.physicalWeightIndex(column: 0, inner: 32, shape: shape) == 32 * 256)
  #expect(Opt0068Contract.physicalWeightIndex(column: 256, inner: 0, shape: shape) == 64 * 256)
}

@Test func sourceOrderCPUContractUsesFP16InputsAndFP32Output() throws {
  let activation: [Float16] = [1, 2, -3, 4, -1, 0.5]
  let weight: [Float16] = [2, -1, 3, 0.5, -2, 4]
  let output = try Opt0068Numerics.cpuReference(
    activation: activation, weight: weight, rows: 2, inner: 3, columns: 2
  )
  #expect(output == [14, -12, 4, -2.5])
  #expect(output.allSatisfy { type(of: $0) == Float.self })
}

@Test func metricsRetainClassesSignedZeroAndULP() throws {
  let reference: [Float] = [-0.0, 1, 2, .infinity, .nan]
  let candidate: [Float] = [+0.0, Float(1).nextUp, 2, .infinity, .nan]
  let metrics = try Opt0068Numerics.compare(candidate: candidate, reference: reference)
  #expect(metrics.compared == 5)
  #expect(metrics.signedZeroMismatchCount == 1)
  #expect(metrics.zeroClassMismatchCount == 0)
  #expect(metrics.finiteClassMismatchCount == 0)
  #expect(metrics.nonFiniteClassMismatchCount == 0)
  #expect(metrics.maximumULP == 1)
}

@Test func adversarialSuitePinsAllRequiredFailureClasses() {
  let fixtures = Opt0068Adversarial.fixtures()
  #expect(Set(fixtures.map(\.kind)) == Set(Opt0068Adversarial.Kind.allCases))
  #expect(fixtures.contains { $0.kind == .longKCancellation && $0.inner == 6_144 })
  #expect(fixtures.contains { $0.kind == .benignTail && $0.rows == 33 })
  #expect(fixtures.allSatisfy { $0.activation.count == $0.rows * $0.inner })
  #expect(fixtures.allSatisfy { $0.weight.count == $0.inner * $0.columns })
}

@Test func authenticationFailsClosedWhenActualFixtureIsAbsent() throws {
  let temporary = FileManager.default.temporaryDirectory
    .appendingPathComponent(UUID().uuidString, isDirectory: true)
  try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: temporary) }
  do {
    _ = try Opt0068Contract.authenticate(
      packageDirectory: temporary,
      fixtureManifest: temporary.appendingPathComponent("actual-m2250.json")
    )
    Issue.record("authentication unexpectedly accepted missing package/fixture data")
  } catch let error as Opt0068Error {
    #expect(error.description.contains("missing revision-7 package manifest"))
  }
}

@Test func sourceBundleHashPinsPathsLengthsAndBytes() throws {
  let temporary = FileManager.default.temporaryDirectory
    .appendingPathComponent(UUID().uuidString, isDirectory: true)
  try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: temporary) }
  try Data("alpha".utf8).write(to: temporary.appendingPathComponent("a.txt"))
  try Data("beta".utf8).write(to: temporary.appendingPathComponent("b.txt"))
  let first = try Opt0068Contract.sourceBundleSHA256(
    root: temporary, relativePaths: ["b.txt", "a.txt"]
  )
  let reordered = try Opt0068Contract.sourceBundleSHA256(
    root: temporary, relativePaths: ["a.txt", "b.txt"]
  )
  #expect(first == reordered)
  try Data("beta!".utf8).write(to: temporary.appendingPathComponent("b.txt"))
  let changed = try Opt0068Contract.sourceBundleSHA256(
    root: temporary, relativePaths: ["a.txt", "b.txt"]
  )
  #expect(changed != first)
}
