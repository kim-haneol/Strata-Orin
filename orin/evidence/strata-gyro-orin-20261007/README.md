# Strata Gyro-M on Jetson AGX Orin

ARM64 포팅, SM87 빌드, 커널 검증, Gyro-M pack 준비, MTP 없는 생성, MTP 생성, OpenAI 호환 HTTP API 요청을 완료했다. 원격 runtime은 ${RUNTIME}이다. 2026-10-07 한국 시간에 수행했고, 종료 후 MAXN0을 유지한다.

기본 소스는 agentionai/Strata rc1의 434e136f73a7da2bfd11cde3bb0aa0acaf1328c6이다. shi3z/Strata-DGX-Spark의 ARM64 변경324b0ef56c84f2d92e9f956354bfe38e031776fa를 이식하고, CMake 충돌을 해결할 때 Gyro의 portable Hadamard 소스를 ARM64와 x86 양쪽에 보존했다. 포팅 소스 커밋은 config/inference-strata-gyro.env에 고정한 bb98d09 계열이다. GGML은 agentionai/llama.cpp 93a24e52cc12687213c0432af41c7fe2ab1a0eda이다. 실제 원격 source revision 및 clean status는 source-verification.json에 있다. 패치는 patches/strata-orin-arm64.patch에 보존했다.

장치 selftest에서 Orin SM87, 16 SM, CUDA driver/runtime13020을 확인했다. CUDA ELF 목록에 SM87 코드가 있다. GDN recurrence, BF16 GEMM, GDN, router, TQ2_T expert CPU/GPU parity 및 Hadamard 검증7개가 모두 통과했다. load-check도0 failures다. 원본 expert 및 PLE 데이터는 유지하고, pack의 dense 호환 변환을 기록했다: 460 tensors converted, 96 exact/364 rounded, 최대 절대 오차0.0144. dense.bin은1,485,688,320바이트다.

Gyro-M은 native TQ2_T/TQ2_T expert48층, 24,576 experts, expert 가중치29.88GiB로 로드됐다. 303개 native tensor 및 Q8_0 token embedding/output head와320,001,536행의 PLE direct I/O를 검증했다. --mmap-experts로 별도의 전체 host expert arena를 만들지 않고, 전체24,576 experts를 GPU cache에 둔다. MTP 서버 INFO에서도 expert_slots24576, arena_mib0을 확인했다. verify 로그는100% VRAM resident zero-doorbell graph를 기록했다. 초기 GPU cache 채우기는 수십 초에서 수분 걸렸다.

MTP는 원본 Qwen/Qwen3.8-Flash-Next revision de4b8e4d43b917e7706784d8bb445c9af86a3540에서 필요한31 tensors, 총5,214,301,696바이트만 받았다. 전체 원본 체크포인트를 받지 않았다. 각 tensor SHA256을 검증하고, upstream mtp_pack.py의 Q2_0 recipe와 mtp_rt.py로 runtime을 생성했다. experts.bin707,788,800바이트, dense.bin116,099,072바이트, draft vocabulary106,299개다. 813MiB의 draft layer와275.7MiB의 draft head를 실제 로드했다.

| 작업 | 입력 tokens | 출력 tokens | MTP 없음 내부 decode tok/s | MTP 내부 decode tok/s | accepted/offered |
|---|---:|---:|---:|---:|---:|
| 한국어 대화 | 28 | 22 | 14.31 | 12.11 | 8/23 |
| 코드 | 31 | 128 | 14.76 | 26.22 | 93/104 |
| 한국어 설명 | 38 | 128 | 13.83 | 19.13 | 64/106 |

위 표는 같은 input token IDs로 수행한 짧은 single-run smoke다. 워밍업 후3회 평균이나2K/512 성능 비교가 아니다. 코드와 설명은128토큰 cap에서 잘렸다. 정식 품질 평가 및 긴 context/Hermes agent 검증은 포함하지 않았다. 입력 cache 재사용은0이었다. MTP 내부 속도는 generated/decode_ms이며 IPC stream 속도와 구분한다.

원본 native pack은 verifier capacity --spec>=2를 요구하고, 원본 --serve는 MTP를 필수로 요구한다. 따라서 MTP 없는 검증은 CLI에서 --spec4 --mtp-max-t1, MTP/lookup 없음으로 한 토큰씩 실행했다. MTP는 resident --serve runner에서 --spec4 --spec-min-p0.5 --suffix-draft0으로 수행했다. context16384, INT8 KV, prefill1024 및 전체 expert cache는 양쪽에서 맞췄지만 CLI와 resident IPC의 실행 및 타이머 정의가 달라 가속비를 확정하지 않는다. MTP smoke의 native child exit는 첫 harness에서 subprocess reference를 지운 뒤 조회해 null로 남았다. wrapper exit0과 완료 요청을 검증했고, 향후 harness는 child reference를 보존하도록 수정했다. null을 성공으로 치환하지 않았다.

HTTP /health, /v1/models 및 /v1/chat/completions를 확인했다. API 요청은200으로 완료됐고, “안녕하세요! 반갑습니다. 😊”를 생성했다. prompt20/output11, cached0, HTTP 전체1.843초다. api-smoke.json에 request와response를 보존했다. SIGINT로 server shutdown을 실행하고 systemd MainPID0, ExecMainStatus0, SubState exited, 포트18092 닫힘을 검증했다. 서비스는 현재 실행 중이 아니다.

실행: scripts/serve-strata-gyro.sh를 Jetson에서 사용하면 원격 server.json 설정으로 MTP 서버를 띄운다. 엔드포인트는 http://127.0.0.1:18092/v1이며 모델명은 qwen38-flash-next-gyro-m이다. Python은 uv를 사용한다. 현재 launcher의 context는16K이며 prompt cache는 검증 조건에 맞춰 꺼져 있다.

ASTOP 원자료는 kernel-validation 및 두 생성 phase에서1초 간격으로 기록했다. telemetry-summary.json의 범위는 phase 전체이며 로드 시간을 포함한다. 두 생성 phase의 GPU peak는99.9%였다. global swap peak는 target-only phase732.25MiB, MTP phase539MiB였고, pressure는 기록된 모든 sample에서normal이었다. GPU power 및 GPU temperature는 이 backend에서 unavailable로 기록됐고0으로 해석하지 않는다. 초기/종료 system summary와 swap/pressure를 보존했다. 최종 ASTOP은 IDLE, GPU0%, 가용 메모리 약52GiB, jobs0이다.

실패 시도도 run/ 안에 보존했다: shallow Git bundle clone 실패(소스 Git archive로 해결), HIP 전용 옵션을 CUDA build에 적용한 configure 실패(옵션 제거), root-owned GPU lock에 쓰기 open 실패(read-only fd의 flock으로 해결), --spec0 native pack smoke 실패(단일토큰 CLI verifier로 해결). 이후 실제 build/prepare/kernel 및 양쪽 smoke의 직접 종료 코드는 모두0이다. upstream의 AVX-2/Q5_K라는 일부 진단 문자열은 ARM64 fallback 및 실제 Q8_0 head를 정확히 표시하지 않으므로, 타입·컴파일·load-check 증거를 우선한다.

원격 동결 증거: ${RUNTIME}/evidence-export (read-only). 88개 파일을 SHA256 대조해 복사했고 로컬 run/ 사본도 files.sha256의 모든 항목과 일치함을 확인했다. 원본/파생 가중치는 Jetson runtime에 유지하고 derived-artifacts.sha256에 해시를 보존한다. 기존 dense 결과와 다른 동결 archive는 수정하지 않았다.
