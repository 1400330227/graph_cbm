import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from graph_cbm.modeling.cbm import build_model
from torchvision.transforms import functional as F
from matplotlib.patches import Rectangle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_resnet_imagenet_preprocess(resize_to=(448, 448)):
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor(),
        transforms.Normalize(mean=target_mean, std=target_std)
    ])
    return preprocess


def create_model(num_classes, relation_classes, n_tasks):
    backbone_name = 'resnet50'
    weights_path = f"save_weights/classification/{backbone_name}-model-best.pth"
    model = build_model(
        target_name=backbone_name,
        num_classes=num_classes,
        relation_classes=relation_classes,
        n_tasks=n_tasks,
        weights_path=weights_path,
    )
    return model


def interpretable():
    model = create_model(25, 19, 20)
    model.eval()
    model.to(device)

    # 1. 图像加载
    image_path = 'data/CUB_200_2011/images/012.Yellow_headed_Blackbird/Yellow_Headed_Blackbird_0003_8337.jpg'

    # 用于模型输入 (PIL Image)
    # 模型内部会处理转换
    try:
        pil_image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"错误: 找不到图像文件 '{image_path}'。请创建一个虚拟图像或提供正确路径。")
        # 创建一个虚拟图像以便代码继续执行
        pil_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

    # 用于可视化 (NumPy array via OpenCV)
    # OpenCV 读取的图像是 BGR 格式的 NumPy 数组，非常适合绘图
    vis_image_rgb = np.array(pil_image)
    image_tensor = F.to_tensor(pil_image).to(device)
    # 2. 模型推理
    with torch.no_grad():
        output_graphs = model([image_tensor])

    # 3. 提取结果
    output_graph = output_graphs[0]
    attention_weights = output_graph['object_attention_weights'].cpu().numpy()
    boxes = output_graph['boxes'].cpu().numpy()

    fig, ax = plt.subplots(1, figsize=(10, 10))

    # Display the base image
    ax.imshow(vis_image_rgb)
    ax.axis('off')  # Hide the axes ticks

    # 5. Draw transparent rectangles on top of the image
    if boxes.shape[0] > 0:
        # Normalize weights to [0, 1] for alpha (transparency)
        min_w, max_w = attention_weights.min(), attention_weights.max()
        norm_weights = (attention_weights - min_w) / (max_w - min_w + 1e-6)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            box_height = y2 - y1

            # Get the attention weight for this box
            alpha = norm_weights[i][0]

            # Create a Rectangle patch
            rect = Rectangle(
                (x1, y1),  # (x,y) bottom-left corner
                box_width,  # Width
                box_height,  # Height
                facecolor='red',  # Fill color
                alpha=alpha,  # Transparency based on attention
                edgecolor='none'  # No border
            )

            # Add the patch to the axes
            ax.add_patch(rect)
    else:
        print("模型没有检测到任何对象。")

    ax.set_title("Attention Heatmap (Matplotlib)")

    # 6. Save the plot to a file (This is the server-safe method)
    output_filename = "attention_heatmap_matplotlib.png"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    print(f"解释性结果已成功保存到文件: {output_filename}")

    # 7. Try to display the plot (This may fail on a server without a display)
    try:
        print("正在尝试显示图像窗口...")
        plt.show()
    except Exception as e:
        print(f"\n无法显示图像窗口 (这是在服务器上的预期行为)。错误: {e}")
        print("请查看已保存的文件。")


if __name__ == '__main__':
    interpretable()
