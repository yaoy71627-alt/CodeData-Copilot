"""Font deployment errors stay explicit and can recover without rerunning Qwen."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pypdf import PdfReader
import pytest
import app
import report
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

BUNDLED = Path(report.__file__).parent / 'assets/fonts/NotoSansSC-Regular.ttf'


def test_missing_custom_font_falls_back_to_bundled(monkeypatch, tmp_path):
    monkeypatch.setenv('CODEDATA_PDF_FONT', str(tmp_path / 'gone.ttf'))
    status = report.pdf_font_status()
    assert status['status'] == 'embedded' and status['source'] == 'bundled'
    assert status['fallback_used'] is True
    assert status['attempts'][0]['error_code'] == 'font_missing'


def test_invalid_custom_font_falls_back_and_keeps_reason(monkeypatch, tmp_path):
    font = tmp_path / 'invalid.ttf'
    font.write_bytes(b'not a font')
    monkeypatch.setenv('CODEDATA_PDF_FONT', str(font))
    status = report.pdf_font_status()
    assert status['status'] == 'embedded' and status['fallback_used']
    assert status['attempts'][0]['error_code'] == 'font_invalid'


def test_relative_custom_font_is_relative_to_project_not_current_directory(monkeypatch, tmp_path):
    monkeypatch.setenv('CODEDATA_PDF_FONT', 'assets/fonts/NotoSansSC-Regular.ttf')
    monkeypatch.chdir(tmp_path)
    status = report.pdf_font_status()
    assert status['status'] == 'embedded' and status['source'] == 'custom'
    assert not status['fallback_used']


def test_missing_bundled_font_is_explicit_and_never_uses_cid_fallback(monkeypatch):
    monkeypatch.delenv('CODEDATA_PDF_FONT', raising=False)
    original = Path.is_file
    monkeypatch.setattr(Path, 'is_file', lambda p: False if p == BUNDLED else original(p))
    status = report.pdf_font_status()
    assert status['status'] == 'unavailable' and status['error_code'] == 'font_missing'
    assert '完整复制' in status['reason'] and 'Markdown' in status['reason']


@pytest.mark.parametrize('fault,expected', [('unreadable', 'font_unreadable'), ('invalid', 'font_invalid'), ('glyphs', 'glyphs_missing')])
def test_bundled_failure_categories(monkeypatch, fault, expected):
    monkeypatch.delenv('CODEDATA_PDF_FONT', raising=False)
    monkeypatch.setattr(pdfmetrics, 'getRegisteredFontNames', lambda: [])
    if fault == 'glyphs':
        replacement = lambda *a, **kw: SimpleNamespace(face=SimpleNamespace(charToGlyph={}))
    else:
        error = PermissionError('private path') if fault == 'unreadable' else ValueError('private parsing detail')
        replacement = Mock(side_effect=error)
    monkeypatch.setattr('reportlab.pdfbase.ttfonts.TTFont', replacement)
    status = report.pdf_font_status()
    assert status['status'] == 'unavailable' and status['error_code'] == expected
    assert 'private' not in status['reason']


def test_cached_font_also_requires_chinese_glyphs(monkeypatch):
    name = report.pdf_font_status()['font_name']
    monkeypatch.setattr(pdfmetrics, 'getFont', lambda key: SimpleNamespace(face=SimpleNamespace(charToGlyph={})))
    assert name in pdfmetrics.getRegisteredFontNames()
    assert report.pdf_font_status()['error_code'] == 'glyphs_missing'


def test_restored_font_refreshes_old_sidebar_status_and_retries_same_report(tmp_path):
    from streamlit.testing.v1 import AppTest
    code = tmp_path / 'model.py'
    code.write_text('value = 1\n')
    state = SimpleNamespace()
    with patch.object(app.st, 'session_state', state), patch.object(app, 'build_pdf_report', side_effect=RuntimeError('PDF 字体缺失')):
        app.analyze_project_files([(code.name, code)], generate_ai=False)
    assert state.report_pdf is None
    before = state.report_data['final_fingerprint']
    markdown = state.report_markdown
    at = AppTest.from_file(str(Path(app.__file__)), default_timeout=40)
    for key,value in vars(state).items():at.session_state[key] = value
    at.session_state['pdf_font_status'] = {'status':'unavailable','reason':'stale font warning'}
    at.session_state['active_page'] = 'Report'
    at.run()
    assert not at.exception
    assert at.session_state['pdf_font_status']['status'] == 'embedded'
    assert not any('stale font warning' in warning.value for warning in at.warning)
    retry = next(button for button in at.button if button.label == '重新生成 PDF')
    retry.click().run()
    assert not at.exception
    assert at.session_state['report_pdf'].startswith(b'%PDF')
    assert at.session_state['report_error'] is None
    assert at.session_state['report_data']['final_fingerprint'] == before
    assert at.session_state['report_markdown'] == markdown


def test_bad_custom_font_still_exports_embedded_chinese_pdf(monkeypatch, tmp_path):
    monkeypatch.setenv('CODEDATA_PDF_FONT', str(tmp_path / 'missing.ttf'))
    final = report.build_report_data({'project_name':'中文字体恢复验证'}, [], [], None, [], '', {})
    final['title'] = '中文字体恢复验证'
    pdf = report.build_pdf_report(final)
    assert b'/FontFile2' in pdf and pdf.startswith(b'%PDF')
    text = ''.join(page.extract_text() for page in PdfReader(BytesIO(pdf)).pages)
    assert '中文字体恢复验证' in text
