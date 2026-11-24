"""用 Streamlit 官方 AppTest 对界面做自动化冒烟测试（无需浏览器）。

运行：
    .venv\\Scripts\\python.exe -m unittest tests.test_ui -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # 未安装 streamlit 时跳过
    AppTest = None

APP = str(ROOT / "app" / "streamlit_app.py")


@unittest.skipIf(AppTest is None, "未安装 streamlit")
class TestStreamlitApp(unittest.TestCase):
    def _run(self, timeout: float = 120.0):
        app = AppTest.from_file(APP, default_timeout=timeout)
        app.run()
        return app

    def test_app_renders_without_exception(self):
        app = self._run()
        self.assertEqual(list(app.exception), [])
        titles = [t.value for t in app.title] + [h.value for h in app.header]
        joined = " ".join(titles)
        self.assertIn("单样本检测", joined)

    def test_switch_pages(self):
        # 每个页面用独立会话渲染，避免 AppTest 在同一会话内反复增删控件导致的内部状态问题
        for page in ["批量检测", "评测", "领域知识库"]:
            app = self._run()
            app.sidebar.radio[0].set_value(page).run()
            self.assertEqual(list(app.exception), [], f"页面 {page} 渲染异常")
            self.assertTrue(app.header or app.title)

    def test_knowledge_page_search(self):
        app = self._run()
        app.sidebar.radio[0].set_value("领域知识库").run()
        self.assertEqual(list(app.exception), [])
        text_inputs = app.text_input
        self.assertTrue(text_inputs, "知识库页应有检索输入框")
        text_inputs[0].set_value("memcpy 越界 长度校验").run()
        self.assertEqual(list(app.exception), [])

    def test_single_detection_end_to_end(self):
        """在界面上真正跑一次离线检测：验证输入 → 事件流 → 四项报告整条链路。"""
        app = self._run()
        sample = (
            "static int copy_name(char *dst, const char *src, int len)\n"
            "{\n"
            "    if (len < 0) return -1;\n"
            "    memcpy(dst, src, len);\n"
            "    return 0;\n"
            "}\n"
        )
        app.text_area[0].set_value(sample).run()
        # 第一个按钮是「从测试集载入」，第二个才是「开始检测」
        detect_buttons = [b for b in app.button if "开始检测" in (b.label or "")]
        self.assertTrue(detect_buttons, "应存在开始检测按钮")
        detect_buttons[0].click().run()
        self.assertEqual(list(app.exception), [])
        markdown_text = " ".join(m.value for m in app.markdown)
        self.assertIn("检测报告", markdown_text)
        self.assertIn("判定理由", " ".join(m.value for m in app.markdown))


if __name__ == "__main__":
    unittest.main(verbosity=2)
