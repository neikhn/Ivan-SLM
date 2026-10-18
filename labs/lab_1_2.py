import torch
from torch import nn

# Giữ cùng dữ liệu và trọng số ban đầu khi chạy lại thí nghiệm
torch.manual_seed(55)

device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

# 1. Dữ liệu cố định: 4 mẫu, mỗi mẫu có 8 đặc trưng.
x = torch.randn(4, 8, device=device)

# Mỗi mẫu có 2 giá trị mục tiêu vì dùng MSE.
y_target = torch.randn(4, 2, device=device)

# 2. Mô hình và công cụ huấn luyện.
layer = nn.Linear(8, 2, bias=False).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(layer.parameters(), lr=0.05)

print("Input shape:", x.shape)
print("Target shape:", y_target.shape)
print("Weight shape:", layer.weight.shape)
print("Gradient ban đầu:", layer.weight.grad)

clear_grad = False

loss_history = []

# 3. Vòng lặp training
for step in range(20):
    if clear_grad: 
        # TODO 1: Xóa gradient cũ bằng optimizer.
        optimizer.zero_grad()
    
    # TODO 2: Truyền x qua layer để lấy dự đoán.
    # y_pred = x @ layer.weight.T
    y_pred = layer(x)

    # TODO 3: Tính loss từ y_pred và y_target bằng loss_fn.
    loss = loss_fn(y_pred, y_target)

    # TODO 4: Tính gradient bằng phương thức của loss.
    loss.backward()

    if step == 0:
        print("Prediction shape:", y_pred.shape)
        print("Gradient sau backward:", layer.weight.grad)
        print("Gradient shape:", layer.weight.grad.shape)

    # TODO 5: Cập nhật trọng số bằng optimizer.
    optimizer.step()

    # TODO 6: Chuyển loss thành số Python để lưu và ghi log.
    loss_value = loss.item()
    loss_history.append(loss_value)

    print(f"Step {step + 1:02d} | Loss: {loss_value:.6f}")

print("Lịch sử loss:", loss_history)

"""
1. weight.grad trước backward: 
    None, vì chưa gọi loss.backward() nên gradient chưa được tính

2. weight.grad.shape sau backward:
    torch.Size([2, 8]).
    Gradient có cùng kích thước với layer.weight:
    2 đầu ra × 8 trọng số cho mỗi đầu ra.

3. Khi có zero_grad():
   Loss đầu = 0.589865
   Loss cuối = 0.049132
   Nhận xét = ...

4. Khi không có zero_grad():
   Loss đầu = 0.589865 
   Loss cuối = 0.283413
   Nhận xét = ...

5. Vì sao dùng loss.item() để ghi log?
    method .item() trong NumPy dùng để lấy 1 giá trị đơn lẻ từ single element tensor hoặc array và chuyển nó về Python datatype như int hoặc float
"""