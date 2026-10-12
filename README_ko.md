# ComfyUI-Filter-Pack

[English](README.md) | [한국어](README_ko.md)

이미지 필터 노드 모음입니다. 색 보정, 샤픈, 디노이즈, 리사이즈를 이미지가 있는 디바이스의 torch 연산으로 처리합니다.

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

## 설치 방법

ComfyUI Manager에서 **ComfyUI-Filter-Pack**을 검색하거나, 아래처럼 설치합니다.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alchemine/comfyui-filter-pack
```

## 노드

`FilterPack/Image`

| 노드 | 설명 |
|------|------|
| **Adjust Image** | 색 보정, 샤픈, 디노이즈, 리사이즈 |

노드 id `AdjustImage`는 comfyui-alchemine-pack이 v8.0.0 전까지 쓰던 것과 같아서, 그 팩으로 만든 워크플로가
그대로 열립니다.

## 라이선스

GPL-3.0. [LICENSE](LICENSE)를 참고하세요.
