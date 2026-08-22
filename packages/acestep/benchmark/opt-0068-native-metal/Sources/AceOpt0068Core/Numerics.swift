import Foundation

public struct NumericMetrics: Codable, Sendable {
  public let compared: Int
  public let finiteClassMismatchCount: Int
  public let nonFiniteClassMismatchCount: Int
  public let signedZeroMismatchCount: Int
  public let zeroClassMismatchCount: Int
  public let maximumULP: UInt32
  public let maximumAbsoluteError: Double
  public let maximumRelativeError: Double
  public let nrmse: Double
  public let snrDB: Double
  public let pearson: Double
}

public enum Opt0068Numerics {
  public static func compare(candidate: [Float], reference: [Float]) throws -> NumericMetrics {
    guard candidate.count == reference.count, !candidate.isEmpty else {
      throw Opt0068Error.contract("numeric comparison arrays differ or are empty")
    }
    var finiteClassMismatchCount = 0
    var nonFiniteClassMismatchCount = 0
    var signedZeroMismatchCount = 0
    var zeroClassMismatchCount = 0
    var maximumULP: UInt32 = 0
    var maximumAbsoluteError = 0.0
    var maximumRelativeError = 0.0
    var squaredError = 0.0
    var squaredReference = 0.0
    var sumCandidate = 0.0
    var sumReference = 0.0
    var sumCandidateSquared = 0.0
    var sumReferenceSquared = 0.0
    var sumProduct = 0.0

    for index in candidate.indices {
      let c = candidate[index]
      let r = reference[index]
      if c.isFinite != r.isFinite { finiteClassMismatchCount += 1 }
      if !c.isFinite || !r.isFinite {
        if !(c.isNaN && r.isNaN) && c != r { nonFiniteClassMismatchCount += 1 }
        continue
      }
      if c == 0, r == 0, c.sign != r.sign { signedZeroMismatchCount += 1 }
      if (c == 0) != (r == 0) { zeroClassMismatchCount += 1 }
      maximumULP = max(maximumULP, ulpDistance(c, r))
      let cd = Double(c)
      let rd = Double(r)
      let error = abs(cd - rd)
      maximumAbsoluteError = max(maximumAbsoluteError, error)
      maximumRelativeError = max(maximumRelativeError, error / max(abs(rd), 1e-30))
      squaredError += error * error
      squaredReference += rd * rd
      sumCandidate += cd
      sumReference += rd
      sumCandidateSquared += cd * cd
      sumReferenceSquared += rd * rd
      sumProduct += cd * rd
    }
    let count = Double(candidate.count)
    let rmsError = sqrt(squaredError / count)
    let rmsReference = sqrt(squaredReference / count)
    let nrmse = rmsError / max(rmsReference, 1e-30)
    let snrDB = 20 * log10(max(rmsReference, 1e-30) / max(rmsError, 1e-30))
    let covariance = count * sumProduct - sumCandidate * sumReference
    let candidateVariance = max(0, count * sumCandidateSquared - sumCandidate * sumCandidate)
    let referenceVariance = max(0, count * sumReferenceSquared - sumReference * sumReference)
    let denominator = sqrt(candidateVariance * referenceVariance)
    let pearson = denominator == 0 ? (candidate == reference ? 1 : 0) : covariance / denominator
    return NumericMetrics(
      compared: candidate.count,
      finiteClassMismatchCount: finiteClassMismatchCount,
      nonFiniteClassMismatchCount: nonFiniteClassMismatchCount,
      signedZeroMismatchCount: signedZeroMismatchCount,
      zeroClassMismatchCount: zeroClassMismatchCount,
      maximumULP: maximumULP,
      maximumAbsoluteError: maximumAbsoluteError,
      maximumRelativeError: maximumRelativeError,
      nrmse: nrmse,
      snrDB: snrDB,
      pearson: pearson
    )
  }

  /// Independent source-K-order FP16-input/FP32-accumulate CPU contract.
  /// This is intentionally scalar and is never part of a performance sample.
  public static func cpuReference(
    activation: [Float16],
    weight: [Float16],
    rows: Int,
    inner: Int,
    columns: Int
  ) throws -> [Float] {
    guard activation.count == rows * inner else {
      throw Opt0068Error.contract("CPU activation length changed")
    }
    guard weight.count == inner * columns else {
      throw Opt0068Error.contract("CPU weight length changed")
    }
    var output = [Float](repeating: 0, count: rows * columns)
    for row in 0..<rows {
      for column in 0..<columns {
        var accumulator: Float = 0
        for k in 0..<inner {
          let product = roundedMultiply(
            Float(activation[row * inner + k]),
            Float(weight[k * columns + column])
          )
          accumulator = roundedAdd(accumulator, product)
        }
        output[row * columns + column] = accumulator
      }
    }
    return output
  }

  public static func hasOnlyFiniteValues(_ values: [Float]) -> Bool {
    values.allSatisfy(\.isFinite)
  }

  private static func ulpDistance(_ left: Float, _ right: Float) -> UInt32 {
    if left == right { return 0 }
    let a = orderedBits(left)
    let b = orderedBits(right)
    return a > b ? a - b : b - a
  }

  private static func orderedBits(_ value: Float) -> UInt32 {
    let bits = value.bitPattern
    return (bits & 0x8000_0000) == 0 ? bits | 0x8000_0000 : ~bits
  }

  @inline(never)
  private static func roundedMultiply(_ left: Float, _ right: Float) -> Float {
    left * right
  }

  @inline(never)
  private static func roundedAdd(_ left: Float, _ right: Float) -> Float {
    left + right
  }
}

public enum Opt0068Adversarial {
  public enum Kind: String, Codable, CaseIterable, Sendable {
    case signedZero = "signed-zero"
    case cancellation
    case finiteRange = "finite-range"
    case longKCancellation = "long-k-cancellation"
    case benignTail = "benign-tail"
  }

  public struct Fixture: Sendable {
    public let kind: Kind
    public let rows: Int
    public let inner: Int
    public let columns: Int
    public let activation: [Float16]
    public let weight: [Float16]
  }

  public static func fixtures() -> [Fixture] {
    [
      make(.signedZero, rows: 33, inner: 32, columns: 256),
      make(.cancellation, rows: 33, inner: 32, columns: 256),
      make(.finiteRange, rows: 33, inner: 32, columns: 256),
      make(.longKCancellation, rows: 3, inner: 6_144, columns: 256),
      make(.benignTail, rows: 33, inner: 2_048, columns: 256),
    ]
  }

  private static func make(_ kind: Kind, rows: Int, inner: Int, columns: Int) -> Fixture {
    var activation = [Float16](repeating: 0, count: rows * inner)
    var weight = [Float16](repeating: 0, count: inner * columns)
    for row in 0..<rows {
      for k in 0..<inner {
        activation[row * inner + k] = Float16(activationValue(kind, row, k))
      }
    }
    for k in 0..<inner {
      for column in 0..<columns {
        weight[k * columns + column] = Float16(weightValue(kind, k, column))
      }
    }
    return Fixture(
      kind: kind, rows: rows, inner: inner, columns: columns,
      activation: activation, weight: weight
    )
  }

  private static func activationValue(_ kind: Kind, _ row: Int, _ inner: Int) -> Float {
    switch kind {
    case .signedZero:
      let values: [Float] = [-0.0, +0.0, -1, 1,
                             -Float(Float16.leastNonzeroMagnitude),
                             Float(Float16.leastNonzeroMagnitude)]
      return values[(inner + row) % values.count]
    case .cancellation:
      let values: [Float] = [2_048, 1, -2_048, 0.5, 1_024, -0.5, -1_024, 1.0 / 1_024]
      return values[(inner + row * 3) % values.count]
    case .finiteRange:
      if inner == 0 { return row.isMultiple(of: 2) ? 65_504 : -65_504 }
      if inner == 1 { return Float(Float16.leastNormalMagnitude) }
      if inner == 2 { return Float(Float16.leastNonzeroMagnitude) }
      return inner.isMultiple(of: 2) ? +0.0 : -0.0
    case .longKCancellation:
      let values: [Float] = [1, 1.0 / 1_024, -1, 1.0 / 2_048, 0.5,
                             -1.0 / 4_096, -0.5, 1.0 / 8_192]
      return values[(inner + row) % values.count]
    case .benignTail:
      return Float(((row * 17 + inner * 13 + 3) % 31) - 15) / 64
    }
  }

  private static func weightValue(_ kind: Kind, _ inner: Int, _ column: Int) -> Float {
    switch kind {
    case .signedZero:
      let values: [Float] = [1, -1, -0.0, +0.0, 0.5, -0.5]
      return values[(inner + column) % values.count]
    case .cancellation:
      return column % 4 < 2 ? 1 : -1
    case .finiteRange:
      if inner == 0 { return [2, -2, 1, 0.5][column % 4] }
      if inner == 1 || inner == 2 { return column.isMultiple(of: 2) ? 1 : -1 }
      return 0
    case .longKCancellation:
      return column % 4 == 0 ? 1 : column % 4 == 1 ? -1 : 0.5
    case .benignTail:
      return Float(((column * 13 + inner * 7 + 7) % 29) - 14) / 64
    }
  }
}
