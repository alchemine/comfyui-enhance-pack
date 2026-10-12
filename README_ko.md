# ComfyUI-Enhance-Pack

[English](README.md) | [한국어](README_ko.md)

이미지 보정 노드 모음입니다. 색 보정, 샤픈, 디노이즈, 점 제거, 리사이즈를 ComfyUI가 계산에 쓰는 GPU의 torch 연산으로 처리합니다.

## 사용법

### Adjust Image

색 보정, 샤픈과 디노이즈 필터, 선택적 리사이즈를 한 노드에 담았습니다. 모든 위젯의 기본값이 무변화라
슬라이더를 움직이기 전까지는 이미지를 그대로 통과시킵니다. RGB만 처리하고 알파 채널은 그대로 보존합니다.

| 위젯 | 하는 일 |
|------|---------|
| `brightness` | 1.0 = 무변화, <1.0 어둡게, >1.0 밝게 |
| `contrast` | 중간 회색 0.5 기준으로 스케일 |
| `saturation` | 0.0 = 흑백, >1.0 = 더 선명하게 |
| `gamma` | <1.0 중간톤 밝게, >1.0 어둡게 |
| `edge_enhance` | 고정 엣지 강화 커널의 블렌드 비율. 0.0 = 끔 |
| `cas` | Contrast Adaptive Sharpening (AMD FidelityFX). 평탄한 영역을 더, 이미 선명한 엣지는 덜 샤픈. 0.0 = 끔 |
| `local_contrast` | 큰 반경 언샤프("clarity")로 중간 스케일의 깊이감 추가. 0.0 = 끔 |
| `denoise` | 엣지 보존 bilateral 필터. 선은 유지하며 평탄 노이즈와 밴딩 제거. 0.0 = 끔 |
| `upscale_method` | `scale_by`가 1.0이 아닐 때의 리샘플링 방식 |
| `scale_by` | 배율. 1.0 = 리사이즈 없음 |

처리 순서는 고정입니다: brightness → contrast → saturation → gamma → denoise → edge enhance → CAS →
local contrast → resize. 디노이즈가 샤픈보다 먼저라, 제거한 노이즈를 샤프너가 증폭하지 않습니다.

### Despeckle Image

생성된 이미지에 흩어진 밝은 점을 지우고, 머리카락 같은 선과 하이라이트, 질감은 그대로 둡니다. 주변보다 밝고, `speck_size`의
두 배 창에 들어가는 덩어리이면서, 주변 패치 안에 그런 덩어리가 자기 말고 몇 개뿐인 것을 점으로 봅니다. 선, 밝은
모서리, 조각 장식, 질감, 옷 주름의 광택은 남습니다. 거의 검은 배경에서는 임계값이 저절로 낮아져 옅은 점도 지워집니다. 점 옆의 어두운 선(윤곽선, 입)은 덮어쓰지 않습니다. 점과 그
어두운 테두리는 둘레 픽셀의 가중 평균으로 채워서, 점 아래의 그라데이션이 끊기지 않습니다.

| 위젯 | 하는 일 |
|------|---------|
| `threshold` | 주변보다 얼마나 밝아야 점으로 보는지. 기본 0.08 |
| `speck_size` | 지울 점의 최대 크기(픽셀). 이보다 넓으면(눈 하이라이트 등) 남깁니다. 기본 5 |
| `mask` (출력) | 바뀐 픽셀. 가장자리는 부드럽게 섞인 값 |

## 설치 방법

ComfyUI Manager에서 **ComfyUI-Enhance-Pack**을 검색하거나, 아래처럼 설치합니다.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alchemine/comfyui-enhance-pack
```

## 노드

`EnhancePack/Image`

| 노드 | 설명 |
|------|------|
| **Adjust Image** | 색 보정, 샤픈, 디노이즈, 리사이즈 |
| **Despeckle Image** | 흩어진 밝은 점 제거 |

노드 id `AdjustImage`는 comfyui-alchemine-pack이 v8.0.0 전까지 쓰던 것과 같아서, 그 팩으로 만든 워크플로가
그대로 열립니다.

## 라이선스

GPL-3.0. [LICENSE](LICENSE)를 참고하세요.
