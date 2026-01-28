"""
ipywidgets 기반 파일/폴더 업로드 위젯
Google Colab 및 Jupyter Notebook에서 사용 가능
"""

import ipywidgets as widgets
from IPython.display import display
import os
import io
import zipfile
import shutil


def create_file_upload_widget(save_path='/content/', accept_zip=True):
    """
    파일 업로드 위젯 생성
    
    Args:
        save_path (str): 파일을 저장할 경로 (기본값: '/content/')
        accept_zip (bool): ZIP 파일 자동 압축 해제 여부 (기본값: True)
    
    Returns:
        widgets.FileUpload: 파일 업로드 위젯
    """
    # 저장 경로 생성
    os.makedirs(save_path, exist_ok=True)
    
    # 출력 위젯 (업로드 상태 표시)
    output = widgets.Output()
    
    # 파일 업로드 위젯 생성
    uploader = widgets.FileUpload(
        accept='',  # 모든 파일 형식 허용
        multiple=True,  # 여러 파일 선택 가능
        description='파일 선택',
        button_style='primary',
        icon='upload'
    )
    
    def on_upload_change(change):
        """파일 업로드 시 실행되는 콜백 함수"""
        with output:
            output.clear_output()
            
            uploaded_files = change['new']
            
            if not uploaded_files:
                print("❌ 업로드된 파일이 없습니다.")
                return
            
            print(f"📦 {len(uploaded_files)}개 파일 처리 중...\n")
            
            for filename, file_info in uploaded_files.items():
                try:
                    # 파일 내용 가져오기
                    content = file_info['content']
                    
                    # ZIP 파일 처리
                    if accept_zip and filename.endswith('.zip'):
                        print(f"📂 '{filename}' - ZIP 파일 압축 해제 중...")
                        
                        # 메모리에서 ZIP 파일 열기
                        with zipfile.ZipFile(io.BytesIO(content), 'r') as zip_ref:
                            # 압축 해제
                            zip_ref.extractall(save_path)
                            
                            # 압축 해제된 파일 목록
                            extracted_files = zip_ref.namelist()
                            print(f"   ✅ {len(extracted_files)}개 파일 압축 해제 완료")
                            
                            # 최상위 폴더 표시
                            if extracted_files:
                                top_items = set(f.split('/')[0] for f in extracted_files)
                                for item in top_items:
                                    print(f"   📁 {save_path}{item}")
                    
                    # 일반 파일 저장
                    else:
                        file_path = os.path.join(save_path, filename)
                        
                        # 파일 쓰기
                        with open(file_path, 'wb') as f:
                            f.write(content)
                        
                        # 파일 크기 계산
                        size = len(content)
                        size_str = f"{size:,} bytes"
                        if size > 1024:
                            size_str = f"{size/1024:.2f} KB"
                        if size > 1024*1024:
                            size_str = f"{size/(1024*1024):.2f} MB"
                        
                        print(f"✅ '{filename}' - {size_str} 저장 완료")
                        print(f"   📂 {file_path}")
                
                except Exception as e:
                    print(f"❌ '{filename}' 처리 중 오류: {str(e)}")
            
            print(f"\n🎉 업로드 완료! 저장 위치: {save_path}")
            print(f"\n📋 저장된 파일 목록:")
            print("-" * 60)
            
            # 저장된 파일 목록 출력
            for item in os.listdir(save_path):
                item_path = os.path.join(save_path, item)
                if os.path.isdir(item_path):
                    print(f"📁 {item}/")
                else:
                    size = os.path.getsize(item_path)
                    print(f"📄 {item} ({size:,} bytes)")
    
    # 파일 업로드 시 콜백 함수 연결
    uploader.observe(on_upload_change, names='value')
    
    # 위젯 레이아웃
    layout = widgets.VBox([
        widgets.HTML(f"<h3>📁 파일 업로드</h3><p>저장 위치: <code>{save_path}</code></p>"),
        uploader,
        output
    ])
    
    return layout


def create_folder_upload_widget(save_path='/content/'):
    """
    폴더 업로드 위젯 (ZIP 파일 전용)
    
    Args:
        save_path (str): 폴더를 저장할 경로 (기본값: '/content/')
    
    Returns:
        widgets.VBox: 폴더 업로드 위젯
    """
    # 저장 경로 생성
    os.makedirs(save_path, exist_ok=True)
    
    # 출력 위젯
    output = widgets.Output()
    
    # ZIP 파일만 허용하는 업로드 위젯
    uploader = widgets.FileUpload(
        accept='.zip',  # ZIP 파일만 허용
        multiple=False,  # 단일 파일만
        description='ZIP 선택',
        button_style='success',
        icon='folder-open'
    )
    
    def on_folder_upload(change):
        """폴더(ZIP) 업로드 시 실행되는 콜백 함수"""
        with output:
            output.clear_output()
            
            uploaded_files = change['new']
            
            if not uploaded_files:
                print("❌ 업로드된 파일이 없습니다.")
                return
            
            for filename, file_info in uploaded_files.items():
                if not filename.endswith('.zip'):
                    print(f"⚠️  '{filename}'은(는) ZIP 파일이 아닙니다.")
                    continue
                
                try:
                    print(f"📦 '{filename}' 압축 해제 중...\n")
                    
                    # 파일 내용 가져오기
                    content = file_info['content']
                    
                    # ZIP 파일 압축 해제
                    with zipfile.ZipFile(io.BytesIO(content), 'r') as zip_ref:
                        # 압축 해제
                        zip_ref.extractall(save_path)
                        
                        # 압축 해제된 파일 목록
                        extracted_files = zip_ref.namelist()
                        
                        print(f"✅ {len(extracted_files)}개 파일 압축 해제 완료!\n")
                        
                        # 폴더 구조 표시
                        print("📂 폴더 구조:")
                        print("-" * 60)
                        
                        # 최상위 항목만 표시
                        top_items = {}
                        for f in extracted_files:
                            parts = f.split('/')
                            if len(parts) > 0:
                                top = parts[0]
                                if top not in top_items:
                                    top_items[top] = []
                                if len(parts) > 1:
                                    top_items[top].append('/'.join(parts[1:]))
                        
                        for folder, files in top_items.items():
                            folder_path = os.path.join(save_path, folder)
                            if os.path.isdir(folder_path):
                                file_count = len([f for f in files if f])
                                print(f"📁 {folder}/ ({file_count} 파일)")
                            else:
                                print(f"📄 {folder}")
                        
                        print(f"\n🎉 업로드 완료! 저장 위치: {save_path}")
                
                except zipfile.BadZipFile:
                    print(f"❌ '{filename}'은(는) 유효한 ZIP 파일이 아닙니다.")
                except Exception as e:
                    print(f"❌ 오류 발생: {str(e)}")
    
    # 콜백 함수 연결
    uploader.observe(on_folder_upload, names='value')
    
    # 위젯 레이아웃
    layout = widgets.VBox([
        widgets.HTML(
            f"<h3>📁 폴더 업로드 (ZIP)</h3>"
            f"<p>저장 위치: <code>{save_path}</code></p>"
            f"<p style='color: #666;'>💡 폴더를 ZIP으로 압축한 후 업로드하세요</p>"
        ),
        uploader,
        output
    ])
    
    return layout


def create_advanced_upload_widget(save_path='/content/'):
    """
    고급 업로드 위젯 (파일/폴더 모두 지원)
    
    Args:
        save_path (str): 저장 경로 (기본값: '/content/')
    
    Returns:
        widgets.VBox: 고급 업로드 위젯
    """
    # 탭 위젯으로 파일/폴더 업로드 구분
    file_widget = create_file_upload_widget(save_path)
    folder_widget = create_folder_upload_widget(save_path)
    
    tab = widgets.Tab(children=[file_widget, folder_widget])
    tab.set_title(0, '📄 파일 업로드')
    tab.set_title(1, '📁 폴더 업로드')
    
    return tab


# ============================================================
# 사용 예시
# ============================================================
if __name__ == "__main__":
    # 예시 1: 기본 파일 업로드
    print("예시 1: 기본 파일 업로드")
    widget1 = create_file_upload_widget(save_path='/content/uploads/')
    display(widget1)
    
    # 예시 2: 폴더 업로드 (ZIP)
    print("\n예시 2: 폴더 업로드")
    widget2 = create_folder_upload_widget(save_path='/content/data/')
    display(widget2)
    
    # 예시 3: 고급 업로드 (탭 형식)
    print("\n예시 3: 고급 업로드")
    widget3 = create_advanced_upload_widget(save_path='/content/')
    display(widget3)
