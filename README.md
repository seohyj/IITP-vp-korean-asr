# 보이스피싱 통화 특화 한국어 음성인식 서버 및 키워드 부스팅 프로그램

## 본 프로그램의 특징

본 프로그램은 보이스피싱 통화 음성을 텍스트로 변환하는 한국어 음성인식(ASR) 서버와 통화 녹음 클라이언트입니다. Whisper-large-v3에 실제 보이스피싱 통화로 도메인 적응한 LoRA 어댑터를 결합하고, 디코딩 시점에 Trie 기반 키워드 부스팅을 적용하여 '수사관', '검찰', '대포통장' 등 보이스피싱 핵심 어휘의 인식률을 높입니다. 단말의 통화 녹음 모듈이 보내는 5초 단위 음성 조각을 HTTP로 받아 인식하며, 인식 결과는 텍스트 기반 보이스피싱 탐지 모델의 입력으로 사용됩니다.

- 통화 음성 3개 평가셋(보이스피싱 통화·합성음·낭독체, 3,103개 발화) 통합 단어오류율(WER) 4.95%
- 모델 재학습 없이 키워드 목록 교체만으로 신규 수법 어휘에 대응

## 주요 기능

### 1. 음성인식 서버 (`asr_server.py`)
- **도메인 적응 모델 서빙**: Whisper-large-v3 + 보이스피싱 LoRA 어댑터를 fp16으로 병합해 적재. 디코딩 설정 language=ko, beam 5, 최대 440 토큰.
- **REST API**: `POST /v1/transcribe`(본문: 오디오 파일 바이트) → `{"text", "silent", "duration_sec", "elapsed_sec"}`, `GET /health`(모델 구성 조회).
- **입력 표준화**: WAV/FLAC 등 입력을 16 kHz 모노로 변환. 다채널 평균, 샘플레이트 자동 변환.
- **무음 처리**: RMS -50 dBFS 미만 조각은 디코딩하지 않음(무음 구간에서 Whisper가 문장을 생성하는 현상 방지).
- **장치 자동 선택**: CUDA → Apple MPS → CPU.

### 2. Trie 키워드 부스팅 (`keyword_boost.py`)
- **디코더 단 문맥 편향**: Whisper `generate()`의 `logits_processor`로 동작하며, 키워드 토큰열의 접두사를 따라가는 후보 토큰의 logit에 가중치를 더함.
- **빔 탐색 호환**: 매 스텝 출력 토큰의 접미사로 Trie 위치를 다시 계산하는 무상태(stateless) 방식이라 빔 재정렬에 영향을 받지 않음.
- **한국어 조사 대응**: 단어 종료를 강제하지 않아 "검거를", "검거가" 등 조사 결합형이 자연스럽게 생성됨. 공백 접두 변이(" 검거")도 함께 등록.
- **키워드 목록**: `keywords/keywords_vp_top100.txt` — 보이스피싱 통화와 일반 전화 통화의 명사 빈도비로 추출한 도메인 키워드 100개.

### 3. 통화 녹음 클라이언트 (`asr_client.py`)
- **통화 조각 분할**: 통화 녹음을 단말과 같은 5초 고정 조각(`--split fixed`) 또는 쉬는 지점 기준 조각(`--split pause`)으로 분할.
- **순차 인식 및 통화 전사문 생성**: 조각을 순서대로 서버에 보내고 시간 정보와 함께 전사문을 조립.
- **자동 평가(선택)**: 정답 텍스트를 주면 문자 오류율(CER, 공백·문장부호 제외)을 계산하고 결과를 JSON으로 저장.

## 설치 및 실행

```bash
pip install -r requirements.txt

# 서버
python asr_server.py \
  --base-model <whisper-large-v3 디렉터리> \
  --adapter <LoRA 어댑터 디렉터리> \
  --keywords keywords/keywords_vp_top100.txt \
  --host 0.0.0.0 --port 8000

# 클라이언트
python asr_client.py call.wav --url http://127.0.0.1:8000 --split pause --ref call_ref.txt --out result.json
```

- 베이스 모델: [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3). LoRA 어댑터 가중치는 저장소에 포함하지 않습니다.
- m4a/mp3 입력은 ffmpeg가 필요합니다.
- 실시간 처리(5초 조각을 5초 안에 인식)에는 CUDA GPU를 권장합니다.

### 실행 예시

```
server: {'status': 'ok', 'base_model': 'whisper-large-v3', 'adapter': 'adapter_vp_r3', 'keywords': 'keywords_vp_top100.txt', 'device': 'mps', 'dtype': 'float16', 'num_beams': 5}
[01]    0.0s  안녕하세요 서울 중앙직원 첨단 범죄 수사팀 김민수 수사관입니다
[02]    5.2s  고객님 명의로 개설된 대포통장이
[03]    7.9s  금융사기 사건에 연루되어 연락드렸습니다
[04]   11.7s  지금부터 통화 내용은 모두 녹취되고 있습니다
[05]   15.0s  본인 확인을 위해 사용하시는 은행과 계좌번호를 말씀해 주시겠습니까
...
CER (spaces/punctuation ignored): 3.12%
```

(가상의 수사기관 사칭 대본을 음성 복제로 합성한 약 30초 통화 예시)

## 구성

```
├── asr_server.py               음성인식 서버 (FastAPI)
├── keyword_boost.py            Trie 키워드 부스팅 LogitsProcessor
├── asr_client.py               통화 녹음 분할·전송·전사문 생성 클라이언트
├── keywords/
│   └── keywords_vp_top100.txt  보이스피싱 도메인 키워드
└── requirements.txt
```
