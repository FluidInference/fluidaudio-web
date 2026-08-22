"""Immutable upstream contracts for the ACE-Step browser packages.

These constants are intentionally ordinary Python data rather than data fetched
from a mutable API. Updating any repository revision, file identity, or tensor
inventory is a reviewed package-format change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


ACE_REPOSITORY = "ACE-Step/Ace-Step1.5"
ACE_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
PLANNER_REPOSITORY = "ACE-Step/acestep-5Hz-lm-0.6B"
PLANNER_REVISION = "148d8ea0225bdab342ee1ae3a354275ccd60ca80"

REFERENCE_REPOSITORY = "https://github.com/ace-step/ACE-Step-1.5.git"
REFERENCE_REVISION = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0"
REFERENCE_LICENSE_GIT_BLOB = "600451d484a555c1273baa2602f32a37fdd0d0ab"
REFERENCE_LICENSE_BYTES = 1_064
REFERENCE_LICENSE_SHA256 = (
    "05a6bce42a62636d2cfb24139cc008b6b899754e244175814bb5dd2f4a485357"
)


@dataclass(frozen=True, slots=True)
class SafetensorsContract:
    """Expected identity of a safetensors header and its tensor inventory."""

    tensor_count: int
    parameter_count: int
    header_length: int
    header_sha256: str
    inventory_sha256: str


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    """One exact file at an immutable Hugging Face repository revision."""

    key: str
    component: str
    repository: str
    revision: str
    path: str
    byte_length: int
    sha256: str
    package_path: str | None = None
    safetensors: SafetensorsContract | None = None

    def __post_init__(self) -> None:
        if len(self.revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.revision
        ):
            raise ValueError(f"{self.key}: revision must be a full lowercase commit")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError(f"{self.key}: invalid SHA-256")
        source_path = PurePosixPath(self.path)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError(f"{self.key}: unsafe repository path {self.path!r}")
        if self.package_path is not None:
            package_path = PurePosixPath(self.package_path)
            if package_path.is_absolute() or ".." in package_path.parts:
                raise ValueError(
                    f"{self.key}: unsafe package path {self.package_path!r}"
                )

    @property
    def resolve_url(self) -> str:
        return (
            f"https://huggingface.co/{self.repository}/resolve/"
            f"{self.revision}/{self.path}"
        )

    def cache_path(self, cache_root: Path) -> Path:
        repository_slug = self.repository.replace("/", "--")
        return cache_root / "source" / repository_slug / self.revision / self.path


MAIN_INVENTORY = SafetensorsContract(
    tensor_count=677,
    parameter_count=2_393_872_518,
    header_length=80_560,
    header_sha256="1b5ffdcb5660c9ca5bf91ad3867afc266b2697ba962e6580fd5675f82ca344f1",
    inventory_sha256=(
        "0befc4e612073b1bd27bee7d98fa119aa96a6a05c9ebd6c1952e4796ab60e33d"
    ),
)
QWEN_INVENTORY = SafetensorsContract(
    tensor_count=310,
    parameter_count=595_776_512,
    header_length=33_384,
    header_sha256="1ff013283b0190994f5557f04301841482bcba6c006dd557fcb15563d4f1768b",
    inventory_sha256=(
        "7b21319da9ddf0ee1bef76ac0b6b8b5afa3ad92d4fffe520416e3904563f3a76"
    ),
)
VAE_INVENTORY = SafetensorsContract(
    tensor_count=365,
    parameter_count=168_695_426,
    header_length=40_528,
    header_sha256="ee65a4b1609150f6be18b8b164c77f340a326c3a5912acc686ea5227f60c9528",
    inventory_sha256=(
        "71b0ccfc5236b9aba8593b5ccece3367714b603f1464a444447e01cc6f467c4b"
    ),
)
PLANNER_INVENTORY = SafetensorsContract(
    tensor_count=310,
    parameter_count=662_884_352,
    header_length=35_312,
    header_sha256="d5edc2f6c305615da41909d7bdf90088d317ccf19ff039cc4406c57abdfb8e0c",
    inventory_sha256=(
        "447349101df0840017aa014a63e2989246caa81430beb9ecb8724feabff5dc96"
    ),
)


SOURCE_ARTIFACTS: tuple[SourceArtifact, ...] = (
    SourceArtifact(
        key="ace-model-card",
        component="licenses",
        repository=ACE_REPOSITORY,
        revision=ACE_REVISION,
        path="README.md",
        byte_length=5_496,
        sha256="69d69b89e313706b93b420b13aa6232259b207a1cd19b0b70358d166e8e451a0",
        package_path="licenses/ACE-Step-Ace-Step1.5-README.md",
    ),
    SourceArtifact(
        key="ace-turbo-config",
        component="dit",
        repository=ACE_REPOSITORY,
        revision=ACE_REVISION,
        path="acestep-v15-turbo/config.json",
        byte_length=1_968,
        sha256="74745ff704ea49164c3d2d1c99fc0670f3fc635a869f0aec2d1311e6a52d400a",
        package_path="assets/ace-turbo-config.json",
    ),
    SourceArtifact(
        key="ace-turbo-weights",
        component="ace",
        repository=ACE_REPOSITORY,
        revision=ACE_REVISION,
        path="acestep-v15-turbo/model.safetensors",
        byte_length=4_787_825_604,
        sha256="3f6e0797fad420a39bd33979eb6e840e30989e34a3794e843d23b60ec6e422d7",
        safetensors=MAIN_INVENTORY,
    ),
    SourceArtifact(
        key="ace-silence-latent",
        component="constants",
        repository=ACE_REPOSITORY,
        revision=ACE_REVISION,
        path="acestep-v15-turbo/silence_latent.pt",
        byte_length=3_841_215,
        sha256="a778e9dd942f5e8b2c09c55370782d318834432b03dabbcdf70e6ed49ad6358b",
    ),
    SourceArtifact(
        key="qwen-config",
        component="text",
        repository=ACE_REPOSITORY,
        revision=ACE_REVISION,
        path="Qwen3-Embedding-0.6B/config.json",
        byte_length=1_359,
        sha256="bb23c1607cfe059a58d8f0196cf1cebb52082b1056b8e358a579da80a5759420",
        package_path="assets/qwen/config.json",
    ),
    SourceArtifact(
        key="qwen-weights",
        component="text",
        repository=ACE_REPOSITORY,
        revision=ACE_REVISION,
        path="Qwen3-Embedding-0.6B/model.safetensors",
        byte_length=1_191_586_416,
        sha256="0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
        safetensors=QWEN_INVENTORY,
    ),
    SourceArtifact(
        key="qwen-tokenizer",
        component="text-tokenizer",
        repository=ACE_REPOSITORY,
        revision=ACE_REVISION,
        path="Qwen3-Embedding-0.6B/tokenizer.json",
        byte_length=11_423_705,
        sha256="def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
        package_path="assets/qwen/tokenizer.json",
    ),
    SourceArtifact(
        key="qwen-tokenizer-config",
        component="text-tokenizer",
        repository=ACE_REPOSITORY,
        revision=ACE_REVISION,
        path="Qwen3-Embedding-0.6B/tokenizer_config.json",
        byte_length=5_404,
        sha256="443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0",
        package_path="assets/qwen/tokenizer_config.json",
    ),
    SourceArtifact(
        key="qwen-merges",
        component="text-tokenizer",
        repository=ACE_REPOSITORY,
        revision=ACE_REVISION,
        path="Qwen3-Embedding-0.6B/merges.txt",
        byte_length=1_671_853,
        sha256="8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        package_path="assets/qwen/merges.txt",
    ),
    SourceArtifact(
        key="qwen-vocab",
        component="text-tokenizer",
        repository=ACE_REPOSITORY,
        revision=ACE_REVISION,
        path="Qwen3-Embedding-0.6B/vocab.json",
        byte_length=2_776_833,
        sha256="ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        package_path="assets/qwen/vocab.json",
    ),
    SourceArtifact(
        key="qwen-added-tokens",
        component="text-tokenizer",
        repository=ACE_REPOSITORY,
        revision=ACE_REVISION,
        path="Qwen3-Embedding-0.6B/added_tokens.json",
        byte_length=707,
        sha256="c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680",
        package_path="assets/qwen/added_tokens.json",
    ),
    SourceArtifact(
        key="qwen-special-tokens",
        component="text-tokenizer",
        repository=ACE_REPOSITORY,
        revision=ACE_REVISION,
        path="Qwen3-Embedding-0.6B/special_tokens_map.json",
        byte_length=613,
        sha256="76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
        package_path="assets/qwen/special_tokens_map.json",
    ),
    SourceArtifact(
        key="qwen-chat-template",
        component="text-tokenizer",
        repository=ACE_REPOSITORY,
        revision=ACE_REVISION,
        path="Qwen3-Embedding-0.6B/chat_template.jinja",
        byte_length=4_116,
        sha256="87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5",
        package_path="assets/qwen/chat_template.jinja",
    ),
    SourceArtifact(
        key="vae-config",
        component="vae",
        repository=ACE_REPOSITORY,
        revision=ACE_REVISION,
        path="vae/config.json",
        byte_length=425,
        sha256="14e019904df567f26df750317a70e2bd08f9f8f3c40ff4a24c97d1cd3f20ccd2",
        package_path="assets/vae-config.json",
    ),
    SourceArtifact(
        key="vae-weights",
        component="vae",
        repository=ACE_REPOSITORY,
        revision=ACE_REVISION,
        path="vae/diffusion_pytorch_model.safetensors",
        byte_length=337_431_388,
        sha256="da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0",
        safetensors=VAE_INVENTORY,
    ),
    SourceArtifact(
        key="planner-model-card",
        component="licenses",
        repository=PLANNER_REPOSITORY,
        revision=PLANNER_REVISION,
        path="README.md",
        byte_length=5_498,
        sha256="ed42093e8f7903f23bd45ea56bea8f484c8aee47820ae8880d6da752681a1da7",
        package_path="licenses/ACE-Step-acestep-5Hz-lm-0.6B-README.md",
    ),
    SourceArtifact(
        key="planner-config",
        component="planner",
        repository=PLANNER_REPOSITORY,
        revision=PLANNER_REVISION,
        path="config.json",
        byte_length=1_386,
        sha256="873ca63808b74e1e008ab950114765553b898b489b9e077e80db83348a118384",
        package_path="assets/planner/config.json",
    ),
    SourceArtifact(
        key="planner-weights",
        component="planner",
        repository=PLANNER_REPOSITORY,
        revision=PLANNER_REVISION,
        path="model.safetensors",
        byte_length=1_325_804_024,
        sha256="5d92a60806e2e88c04de58ddc6dde93f2bc8f1336162b3ad5853886c9bcc6b82",
        safetensors=PLANNER_INVENTORY,
    ),
    SourceArtifact(
        key="planner-tokenizer",
        component="planner-tokenizer",
        repository=PLANNER_REPOSITORY,
        revision=PLANNER_REVISION,
        path="tokenizer.json",
        byte_length=24_321_939,
        sha256="35af56c3f5cb3ea2cc578aa28a8937770981d504f183ac5c8c38baf4bbd4af4d",
        package_path="assets/planner/tokenizer.json",
    ),
    SourceArtifact(
        key="planner-tokenizer-config",
        component="planner-tokenizer",
        repository=PLANNER_REPOSITORY,
        revision=PLANNER_REVISION,
        path="tokenizer_config.json",
        byte_length=14_072_925,
        sha256="6cd70cdd89425971794f5235562edcc608b0629a6c4686ae51a8b8c8b8ba5e95",
        package_path="assets/planner/tokenizer_config.json",
    ),
    SourceArtifact(
        key="planner-merges",
        component="planner-tokenizer",
        repository=PLANNER_REPOSITORY,
        revision=PLANNER_REVISION,
        path="merges.txt",
        byte_length=1_671_853,
        sha256="8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        package_path="assets/planner/merges.txt",
    ),
    SourceArtifact(
        key="planner-vocab",
        component="planner-tokenizer",
        repository=PLANNER_REPOSITORY,
        revision=PLANNER_REVISION,
        path="vocab.json",
        byte_length=2_776_833,
        sha256="ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        package_path="assets/planner/vocab.json",
    ),
    SourceArtifact(
        key="planner-added-tokens",
        component="planner-tokenizer",
        repository=PLANNER_REPOSITORY,
        revision=PLANNER_REVISION,
        path="added_tokens.json",
        byte_length=2_217_787,
        sha256="db08b66a515fb5d6acca0b3492d25bb44e0deda6241fc1113ac0679d40558c48",
        package_path="assets/planner/added_tokens.json",
    ),
    SourceArtifact(
        key="planner-special-tokens",
        component="planner-tokenizer",
        repository=PLANNER_REPOSITORY,
        revision=PLANNER_REVISION,
        path="special_tokens_map.json",
        byte_length=1_824_199,
        sha256="76e233bfc357b0b03d7b6e6ba8e799244f358552575a7cdadb78d3d19106b298",
        package_path="assets/planner/special_tokens_map.json",
    ),
    SourceArtifact(
        key="planner-chat-template",
        component="planner-tokenizer",
        repository=PLANNER_REPOSITORY,
        revision=PLANNER_REVISION,
        path="chat_template.jinja",
        byte_length=4_168,
        sha256="a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        package_path="assets/planner/chat_template.jinja",
    ),
)

ARTIFACT_BY_KEY = {artifact.key: artifact for artifact in SOURCE_ARTIFACTS}
if len(ARTIFACT_BY_KEY) != len(SOURCE_ARTIFACTS):
    raise RuntimeError("Source artifact keys must be unique")

SAFETENSOR_ARTIFACTS = tuple(
    artifact for artifact in SOURCE_ARTIFACTS if artifact.safetensors is not None
)
EXPECTED_SAFETENSOR_COUNT = 1_662
if sum(
    artifact.safetensors.tensor_count
    for artifact in SAFETENSOR_ARTIFACTS
    if artifact.safetensors is not None
) != EXPECTED_SAFETENSOR_COUNT:
    raise RuntimeError("Safetensor inventory total is internally inconsistent")
