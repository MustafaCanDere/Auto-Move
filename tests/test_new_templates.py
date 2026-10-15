import pytest
import visionflow.nodes  # noqa: F401 - registers built-in nodes
from visionflow import templates as T
from visionflow.core.graph import Graph

NEW_TEMPLATES_TO_TEST = [
    "dijital_pasaport_ve_akilli_rapor",
    "yapay_zeka_vlm_ve_kara_kutu",
    "robot_rehberlik_ve_spc_istatistik",
    "artirilmis_gerceklik_montaj_rehberi",
    "3d_lazer_profil_ve_konveyor_ayiklama",
    "cobot_guvenlik_bariyeri_ve_dpm_barkod",
]


def test_gallery_lists_all_new_templates():
    items = T.list_templates()
    names = [t["name"] for t in items]
    for name in NEW_TEMPLATES_TO_TEST:
        assert name in names, f"Template {name} missing from list_templates()"


def test_new_templates_metadata_and_categories():
    items = {t["name"]: t for t in T.list_templates()}
    for name in NEW_TEMPLATES_TO_TEST:
        t = items[name]
        assert t["title"], f"{name} has no title"
        assert t["description"], f"{name} has no description"
        assert t["categories"], f"{name} has no categories"
        for cat in t["categories"]:
            assert cat in T.CATEGORIES, f"{cat} in {name} is not in T.CATEGORIES"


@pytest.mark.parametrize("name", NEW_TEMPLATES_TO_TEST)
def test_new_template_graph_builds_and_runs(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    items = {t["name"]: t for t in T.list_templates()}
    t = items[name]

    # Build Graph using apply_answers
    g = T.apply_answers(t)
    assert isinstance(g, Graph)
    assert len(g.nodes) >= 3

    # Run 1 frame to ensure no crash
    errors = []

    def on_frame(n, results):
        for nid, r in results.items():
            if r.error:
                errors.append((nid, r.error))

    frames = g.run(max_frames=1, on_frame=on_frame)
    assert frames == 1
    assert not errors, f"Errors in template {name}: {errors}"
