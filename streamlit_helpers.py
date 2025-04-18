import io
import sys
from contextlib import contextmanager

class StreamToStreamlit(io.StringIO):
    """Redirects stdout to a Streamlit container with colorized logs."""
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.log = ""
    def write(self, s):
        self.log += s
        self._render_log()
        return len(s)
    def flush(self):
        pass
    def _render_log(self):
        html_log = ""
        for line in self.log.splitlines():
            if "ERROR" in line:
                html_log += f'<div style="color:#ff4b4b;">{line}</div>'
            elif "WARNING" in line:
                html_log += f'<div style="color:#ffa500;">{line}</div>'
            elif "INFO" in line:
                html_log += f'<div style="color:#1e90ff;">{line}</div>'
            else:
                html_log += f'<div style="color:#d3d3d3;">{line}</div>'
        self.container.markdown(
            f'''<div style="height:350px;overflow-y:auto;background:#181818;padding:8px;border-radius:6px;font-size:13px;">{html_log}</div>''',
            unsafe_allow_html=True
        )

def render_log_to_streamlit(log_container, log_text):
    """Render log text to a Streamlit container with colorization."""
    html_log = ""
    for line in log_text.splitlines():
        if "ERROR" in line:
            html_log += f'<div style="color:#ff4b4b;">{line}</div>'
        elif "WARNING" in line:
            html_log += f'<div style="color:#ffa500;">{line}</div>'
        elif "INFO" in line:
            html_log += f'<div style="color:#1e90ff;">{line}</div>'
        else:
            html_log += f'<div style="color:#d3d3d3;">{line}</div>'
    log_container.markdown(
        f'''<div style="height:350px;overflow-y:auto;background:#181818;padding:8px;border-radius:6px;font-size:13px;">{html_log}</div>''',
        unsafe_allow_html=True
    )

@contextmanager
def redirect_stdout_to_streamlit(container):
    """Context manager to redirect stdout to Streamlit log container."""
    stream = StreamToStreamlit(container)
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout

@contextmanager
def capture_stdout():
    """Context manager to capture stdout into a StringIO buffer."""
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        yield buffer
    finally:
        sys.stdout = old_stdout
