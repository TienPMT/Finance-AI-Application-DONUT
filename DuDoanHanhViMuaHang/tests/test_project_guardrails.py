import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_FILE_HASHES = {
    ".claude/reference/.gitignore": "64e15ffad9a9ec51f5204349a13b7862cf3465161571c9d10ec6835972b23d6a",
    ".claude/reference/CMSpade/.gitignore": "b1294ddfd0abee44b20f9af393f759ae82233f8a13c697fff1351cbdd5a2ebd0",
    ".claude/reference/CMSpade/.idea/.gitignore": "1bfdece3645ed8ed356030f22cc2004dc3f401ff060ac3d24de811c3bcd82e16",
    ".claude/reference/CMSpade/.idea/misc.xml": "da8e0070b2c9ef64906d0334b4f9b969652ecec3d3c8121dc0c41601b2b4b8ad",
    ".claude/reference/CMSpade/.idea/modules.xml": "cdf3a688c3e6ddd66369bf29d7f69431ebe3273576ae98c7268a8dd25f4b087e",
    ".claude/reference/CMSpade/CMSpade.iml": "c08ee29851502ea1fc369dcfd51a269deaef93a03d815ff25fd7e1faf12b7d32",
    ".claude/reference/CMSpade/output.txt": "70c8f1c8b78e48c2aa08fae46324335621712d01ae0e9b495e18898e549943cd",
    ".claude/reference/CMSpade/src/CMSpade.java": "32ea987d924327d4a16f1f32cce54f0f074710b868d1208fb35f28674cd30de0",
    ".claude/reference/CMSpade/src/algorithms/AlgoCMSPADE.java": "5a88c6c5656f97632a7bcfc0346b274e08b06b2c32d458e656cfd636564baa2a",
    ".claude/reference/CMSpade/src/algorithms/EquivalenceClass.java": "23eb93e78c39daa88c18f86ee33279d791c32a3cdc35ceaf6441c3ff25589eaf",
    ".claude/reference/CMSpade/src/algorithms/FrequentPatternEnumeration.java": "21485243e4d429d98dd3719a39f29e93aca2d2551a181c4f6ca25b20d706fcd3",
    ".claude/reference/CMSpade/src/algorithms/FrequentPatternEnumerationFacade.java": "5d5cf6aec7a5276e5ef7b2f4bf832bd4a88618639af2c1144c522489bbbd220b",
    ".claude/reference/CMSpade/src/algorithms/candidatePatternsGeneration/CandidateGenerator.java": "f0824cd6ebe3f3afa51430b5fe1df29654f79e7d168cf489bd62b75e2473dd5b",
    ".claude/reference/CMSpade/src/algorithms/candidatePatternsGeneration/CandidateGenerator_Qualitative.java": "9caf0453fdccb563427f28ba9212e8005e59d3824940c4814d94a0f4e1871483",
    ".claude/reference/CMSpade/src/algorithms/dataStructures/Item.java": "dbe3ac767ee875557a5b493d89f284640ed6758cd4f000898a5ef455adc8a112",
    ".claude/reference/CMSpade/src/algorithms/dataStructures/ItemFactory.java": "3ad893e24bbf526707b3d61ad5f96575c461787bc8ba4484a2298999de5f4935",
    ".claude/reference/CMSpade/src/algorithms/dataStructures/Itemset.java": "f595452da398e496302e783df611a92a43b17cd50b2abed5ffb60225b4e2f75e",
    ".claude/reference/CMSpade/src/algorithms/dataStructures/Sequence.java": "bc40189d9e78da97f1c2f0d9244ae9fc60253f083a50372274d506aa58bf5fef",
    ".claude/reference/CMSpade/src/algorithms/dataStructures/Sequences.java": "e05529d99b2307bb8b1caaa924f752caec32f62cc09c7595652c4d6c0369eb5e",
    ".claude/reference/CMSpade/src/algorithms/dataStructures/abstractions/Abstraction_Generic.java": "4f2541ff5f047a25c0db8e9d4fe25ec15def14cd00ae5f1683d4ea3897324a20",
    ".claude/reference/CMSpade/src/algorithms/dataStructures/abstractions/Abstraction_Qualitative.java": "3a5a7c825cb131fefedcc8a5f6de415bef533fa7afb3ee67d1f930fcd5ebb446",
    ".claude/reference/CMSpade/src/algorithms/dataStructures/abstractions/ItemAbstractionPair.java": "0072159571f2da04a34652ee481008bc4f0132c9b01c529cb1df4006626dd589",
    ".claude/reference/CMSpade/src/algorithms/dataStructures/creators/AbstractionCreator.java": "ec9785781ac11a46a4703723252fb74b31a2fe50abf6738e7598a67bc1fe4550",
    ".claude/reference/CMSpade/src/algorithms/dataStructures/creators/AbstractionCreator_Qualitative.java": "a49314b1793f9eab86e363be0b3b9c440d0f0d3fc589d61031680f47bc4a2056",
    ".claude/reference/CMSpade/src/algorithms/dataStructures/creators/ItemAbstractionPairCreator.java": "8ec2ca82ace8523465ca11ca7b576741fcbd5372ddf12d26faf7c9374fa513e9",
    ".claude/reference/CMSpade/src/algorithms/dataStructures/database/SequenceDatabase.java": "c0822a934cab4b8d914a6aafd00fa6a2b60f8b464ec1e233366a47fcb48c27a6",
    ".claude/reference/CMSpade/src/algorithms/dataStructures/patterns/Pattern.java": "518215eee33c5555cb69ec359bf5f9d39c863e6c9e95921483473650fb85682d",
    ".claude/reference/CMSpade/src/algorithms/dataStructures/patterns/PatternCreator.java": "c1367ddd364c3b7f407b3a8aede77bd919a29605be50969b721b1d0893b491c1",
    ".claude/reference/CMSpade/src/algorithms/idLists/IDList.java": "09845b471b92899ffc6955df4440089eb1902e18a4585c6df0f6ea6ca6d09dc3",
    ".claude/reference/CMSpade/src/algorithms/idLists/IDListBitmap.java": "2513740356c3c30601bb06acbb049fbba92591ac943e2931987cc84476ba31f4",
    ".claude/reference/CMSpade/src/algorithms/idLists/IDListFatBitmap.java": "6f9b4e7922af7cb0143c8f8b17c01afc19686b1d3a6d75c3e177b8108912376b",
    ".claude/reference/CMSpade/src/algorithms/idLists/IDListStandard_Map.java": "ca6f63b4859d1a5944d681df75bd7aab0bfc17304cf946e8dc03d1be9b5adeff",
    ".claude/reference/CMSpade/src/algorithms/idLists/creators/IdListCreator.java": "79dff743cc794bab1c3f9fd66ff072c422c5d77ac5a79e39319c8d90c5102886",
    ".claude/reference/CMSpade/src/algorithms/idLists/creators/IdListCreator_Bitmap.java": "e6450a227eb0e81dc6f71bcff94bf537128cf7cb5dc926f1429abcda666108ae",
    ".claude/reference/CMSpade/src/algorithms/idLists/creators/IdListCreator_FatBitmap.java": "7e4f2ff48714f4e3940ab1cefd190def091bbac39f8859e5291ce66fcff933c5",
    ".claude/reference/CMSpade/src/algorithms/idLists/creators/IdListCreator_StandardMap.java": "e9a844f64024c91ae36a43e7bc655dc0d195e48554655f6c4ae5649d7f9f1860",
    ".claude/reference/CMSpade/src/algorithms/savers/Saver.java": "5547228f506f08d03267f4281ec540d4471b21aa62a6bafe5aa37c682f874f11",
    ".claude/reference/CMSpade/src/algorithms/savers/SaverIntoFile.java": "0c4f58a0ca4a4454967b76ef0196bbf38b2ed6d64c0ce86a5d2be19c91f2f67a",
    ".claude/reference/CMSpade/src/algorithms/savers/SaverIntoMemory.java": "01c617e43d343f82cfe28a8358039bb737606aab8a3e06dab4e0ab55d3f10a85",
    ".claude/reference/CMSpade/src/contextPrefixSpan.txt": "87b0883df5acb987356cc781845dc2604c122878aa44e24bef079a34b5d96b85",
    ".claude/reference/CMSpade/src/tools/MemoryLogger.java": "887536ee4ae257c0d6fb4bc17273950beffa31638a20d8ec9bb6eb1851527c99",
    ".claude/reference/paper/2209.02892v2.pdf": "2f52e34731f88dec57ee825d44dfa12023bf2cee6002bf3a9ed5f4c415cafefe",
    ".claude/reference/paper/2330-Article Text-2305-1-10-20180901.pdf": "1e0e00a1e64c026b36c9cfe80bf7c3e95f4b1f49ccc27fc5f53a7e9b7d33f492",
    ".claude/reference/paper/CMSPADE.pdf": "86aabf4dcd2f537ea349080c0b29844ba68427e7b7f23be4f6d2232d91d0f3d0",
    ".claude/reference/paper/ICT2026_WCM_SPADE.pdf": "04633868f1b7c1f749022962023852603f66561915a4a075b03d97cbb4a58db2",
    ".claude/reference/paper/Mining_weighted_sequential_patterns_based_on_prefix_tree_and_prism_encoding_2023.pdf": "b1f019867fd61e21f4e951f464a5cbe79b3917e09cf99dd3b67c8ab18c119c41",
    ".claude/reference/paper/SPADE.pdf": "b64fa7a3b015fb333a065b2d9dcef51f42e5d4be2c7fee420c1a544c1a42d4c4",
}


def readme_required_doc_paths() -> list[str]:
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    must_read_section = readme_text.split("## Must Read First", maxsplit=1)[1].split(
        "## Hard Rules",
        maxsplit=1,
    )[0]
    return re.findall(r"`(\.claude/docs/[^`]+\.md)`", must_read_section)


def test_required_readme_docs_exist() -> None:
    required_doc_paths = readme_required_doc_paths()
    missing_paths = [path for path in required_doc_paths if not (ROOT / path).is_file()]

    assert required_doc_paths != []
    assert missing_paths == []


def test_readme_does_not_require_removed_chapter_six_draft() -> None:
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "chuong-6-thiet-ke-va-cai-dat-he-thong.md" not in readme_text


def git_paths(*args: str) -> list[Path]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / path for path in result.stdout.splitlines() if path]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_only_readme_markdown_exists_outside_claude_directory() -> None:
    markdown_files = git_paths("ls-files", "--", "*.md")
    markdown_files.extend(git_paths("ls-files", "--others", "--exclude-standard", "--", "*.md"))
    invalid_markdown_files = [
        path
        for path in markdown_files
        if path.relative_to(ROOT).parts[0] != ".claude" and path.name != "README.md"
    ]

    assert invalid_markdown_files == []


def test_planned_packages_are_importable() -> None:
    import app
    import app.main
    import mining

    assert app.__name__ == "app"
    assert app.main.app.title == "KPDL Product Recommendation Demo"
    assert mining.__name__ == "mining"


def test_reference_directory_has_no_uncommitted_changes() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", ".claude/reference"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""


def test_tracked_reference_files_match_baseline_hashes() -> None:
    tracked_paths = {
        str(path.relative_to(ROOT))
        for path in git_paths("ls-files", "--", ".claude/reference")
    }
    actual_hashes = {
        path: file_sha256(ROOT / path)
        for path in REFERENCE_FILE_HASHES
    }

    assert tracked_paths == set(REFERENCE_FILE_HASHES)
    assert actual_hashes == REFERENCE_FILE_HASHES
