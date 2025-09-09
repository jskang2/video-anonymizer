# WSL2 + Docker Desktop 가이드

## 전제
- Windows 11/10 + WSL2 + Docker Desktop.
- 리눅스 배포판(예: Ubuntu 22.04)에서 개발/빌드/실행.

## 1) WSL2 확인/업데이트 (PowerShell)
```powershell
wsl -l -v
wsl --set-default-version 2
wsl --update
```

## 2) Docker Desktop 설정(Windows)
- Settings → General: "Use the WSL 2 based engine" 체크
- Settings → Resources → WSL Integration: 사용 중인 배포판(예: Ubuntu)에 체크

## 3) 프로젝트 위치(매우 중요)
- 반드시 WSL 내부 리눅스 파일시스템에 두세요: `/home/<user>/video-anonymizer-mvp`
- C:/Users/... 같은 Windows 경로는 바인드 마운트가 느립니다.

## 4) 클론 & 빌드 (WSL 셸)
```bash
cd ~
# git clone <repo-url> video-anonymizer-mvp
cd ~/video-anonymizer-mvp
make build
```

## 5) 실행 (WSL 셸)
```bash
make demo
make run IN=data/in.mp4 OUT=data/out.mp4 PARTS=eyes,elbows STYLE=mosaic
```

## 6) GPU 가속(선택)
```bash
docker build -f Dockerfile.gpu -t video-anonymizer-mvp:gpu .
docker run --rm --gpus all -v $(pwd):/app video-anonymizer-mvp:gpu \
  python -m anonymizer.cli --input data/in.mp4 --output data/out.mp4 \
  --parts eyes,elbows --style mosaic
```
