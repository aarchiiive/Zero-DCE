import torch
# from torchprofile import profile_macs  # MACs 계산 라이브러리
import model

# 모델 로드
DCE_net = model.enhance_net_nopool().cuda()

# 입력 크기 정의 (예: 256x256 이미지)
# input_size = (720, 1080)  # (높이, 너비)
# input_size = (832, 658) # Exdark
input_size = (900, 1600) # nuImages

dummy_input = torch.randn(1, 3, *input_size).cuda()  # 배치 크기 1, 채널 3

# (옵션) warm-up: GPU 초기화를 위해 몇 번 미리 실행
for _ in range(10):
    _ = DCE_net(dummy_input)

# 100번의 추론 실행 후, 각 추론 시간을 측정하여 저장
times = []
for _ in range(100):
    torch.cuda.synchronize()  # 이전 작업들이 모두 완료되었는지 동기화
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()   # 시작 시간 기록
    _ = DCE_net(dummy_input)  # 추론 실행
    end_event.record()     # 종료 시간 기록

    torch.cuda.synchronize()  # 모든 GPU 연산이 끝날 때까지 대기
    elapsed_time_ms = start_event.elapsed_time(end_event)  # 경과 시간 (밀리초 단위)
    times.append(elapsed_time_ms)

# 평균 추론 시간 계산
avg_time = sum(times) / len(times)
print("Average inference time over 100 runs: {:.3f} ms".format(avg_time))
