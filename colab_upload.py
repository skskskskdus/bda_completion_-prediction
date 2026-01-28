"""
Google Colab 노트북에 직접 복사해서 사용하는 파일 업로드 코드
노트북 셀에 이 코드를 복사하여 실행하세요
"""

# ============================================================
# 코랩 노트북 셀에 복사할 코드 - 기본 파일 업로드
# ============================================================

"""
import ipywidgets as widgets
from IPython.display import display
import os
import io
import zipfile

# 출력 위젯
output = widgets.Output()

# 파일 업로드 위젯
uploader = widgets.FileUpload(
    accept='',  # 모든 파일 형식
    multiple=True,  # 여러 파일 선택 가능
    description='파일 선택',
    button_style='primary',
    icon='upload'
)

def on_upload(change):
    with output:
        output.clear_output()
        uploaded_files = change['new']
        
        if not uploaded_files:
            print("❌ 업로드된 파일이 없습니다.")
            return
        
        save_path = '/content/'
        print(f"📦 {len(uploaded_files)}개 파일 처리 중...\\n")
        
        for filename, file_info in uploaded_files.items():
            content = file_info['content']
            
            # ZIP 파일 처리
            if filename.endswith('.zip'):
                print(f"📂 '{filename}' - ZIP 파일 압축 해제 중...")
                with zipfile.ZipFile(io.BytesIO(content), 'r') as zip_ref:
                    zip_ref.extractall(save_path)
                    print(f"   ✅ 압축 해제 완료: {save_path}")
            else:
                # 일반 파일 저장
                file_path = os.path.join(save_path, filename)
                with open(file_path, 'wb') as f:
                    f.write(content)
                size_mb = len(content) / (1024*1024)
                print(f"✅ '{filename}' - {size_mb:.2f} MB 저장 완료")
        
        print(f"\\n🎉 업로드 완료!\\n")
        print("📋 저장된 파일:")
        !ls -lh /content/

uploader.observe(on_upload, names='value')

# 위젯 표시
display(widgets.VBox([
    widgets.HTML("<h3>📁 파일 업로드</h3>"),
    uploader,
    output
]))
"""


# ============================================================
# 코랩 노트북 셀에 복사할 코드 - 간단 버전
# ============================================================

"""
import ipywidgets as widgets
from IPython.display import display
import zipfile
import io

uploader = widgets.FileUpload(accept='', multiple=True)

def upload(change):
    for name, file in change['new'].items():
        if name.endswith('.zip'):
            zipfile.ZipFile(io.BytesIO(file['content'])).extractall('/content/')
            print(f"✅ {name} 압축 해제 완료")
        else:
            with open(f'/content/{name}', 'wb') as f:
                f.write(file['content'])
            print(f"✅ {name} 저장 완료")
    print("\\n📂 저장된 파일:")
    !ls -lh /content/

uploader.observe(upload, names='value')
display(uploader)
"""


# ============================================================
# 코랩 노트북 셀에 복사할 코드 - 폴더 업로드 (ZIP)
# ============================================================

"""
import ipywidgets as widgets
from IPython.display import display
import zipfile
import io
import os

output = widgets.Output()
uploader = widgets.FileUpload(
    accept='.zip',
    multiple=False,
    description='ZIP 선택',
    button_style='success'
)

def upload_folder(change):
    with output:
        output.clear_output()
        for filename, file_info in change['new'].items():
            if filename.endswith('.zip'):
                print(f"📦 '{filename}' 압축 해제 중...\\n")
                content = file_info['content']
                with zipfile.ZipFile(io.BytesIO(content), 'r') as zip_ref:
                    zip_ref.extractall('/content/')
                    files = zip_ref.namelist()
                    print(f"✅ {len(files)}개 파일 압축 해제 완료!\\n")
                    print("📂 폴더 구조:")
                    !tree /content/ -L 2 || ls -la /content/

uploader.observe(upload_folder, names='value')

display(widgets.VBox([
    widgets.HTML("<h3>📁 폴더 업로드 (ZIP)</h3><p>폴더를 ZIP으로 압축 후 업로드하세요</p>"),
    uploader,
    output
]))
"""
