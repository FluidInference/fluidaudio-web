// swift-tools-version: 6.2

import PackageDescription

let package = Package(
  name: "AceOpt0068NativeMetal",
  platforms: [.macOS(.v15)],
  products: [
    .library(name: "AceOpt0068Core", targets: ["AceOpt0068Core"]),
    .executable(name: "ace-opt-0068-mps", targets: ["AceOpt0068MPS"]),
  ],
  targets: [
    .target(name: "AceOpt0068Core"),
    .executableTarget(
      name: "AceOpt0068MPS",
      dependencies: ["AceOpt0068Core"],
      linkerSettings: [
        .linkedFramework("Metal"),
        .linkedFramework("MetalPerformanceShaders"),
        .linkedFramework("Accelerate"),
      ]
    ),
    .testTarget(
      name: "AceOpt0068CoreTests",
      dependencies: ["AceOpt0068Core"]
    ),
  ]
)
