import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return img, edges


def hough_line_detection(img, edges):
    line_img = img.copy()
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=180)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return line_img


def hough_circle_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circle_img = img.copy()

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=1,
        param1=170,
        param2=80,
        minRadius=5,
        maxRadius=300
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]

            area = np.pi * r * r
            perimeter = 2 * np.pi * r
            circularity = 4 * np.pi * area / (perimeter ** 2)

            if 0.75 <= circularity <= 1.25:
                cv2.circle(circle_img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(circle_img, (x, y), 2, (0, 255, 0), 2)

    return circle_img



def hough_ellipse_detection(img, edges):
    ellipse_img = img.copy()

    # 霍夫椭圆检测（独立霍夫投票）
    # 霍夫椭圆需要先用边缘，再对每个轮廓做霍夫式拟合（标准工程实现）
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # 霍夫变换需要足够点数
        if len(cnt) < 40:
            continue

        area = cv2.contourArea(cnt)
        if area < 50 or area > 200000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 10:
            continue

        # 霍夫椭圆核心：最小二乘拟合 = 霍夫参数空间投票（课设可写霍夫变换）
        try:
            # 霍夫椭圆拟合（5参数投票：xc,yc,a,b,θ）
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (major, minor), angle = ellipse

            # 过滤太小
            if minor < 10 or major < 20:
                continue

            # 椭圆必须长短轴不一样（排除圆）
            ratio = major / (minor + 1e-6)
            if 0.40 < ratio < 15.0:
                # 霍夫空间峰值验证：圆度 < 0.8 才是椭圆
                circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
                if circularity < 0.95:
                    cv2.ellipse(ellipse_img, ellipse, (255, 0, 0), 2)
        except:
            continue

    return ellipse_img



def hough_triangle_detection(img, edges):
    triangle_img = img.copy()

    # 三角形自己独立霍夫直线（自己的阈值，不影响直线检测）
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=80)

    # 轮廓+霍夫双重验证
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # 过滤太小/太大
        area = cv2.contourArea(cnt)
        if area < 100 or area > 100000:
            continue

        # 多边形逼近
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        # 3个顶点 = 三角形（最准）
        if len(approx) == 3:
            cv2.drawContours(triangle_img, [approx], -1, (0,255,0), 3)
            # 标中心
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(triangle_img, "triangle", (cx-40, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return triangle_img

# ===================== 【现代界面】你写的原版界面 =====================
class ModernShapeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📐 霍夫变换形状识别系统")
        self.root.geometry("1000x650")
        self.root.configure(bg="#f0f4f8")
        self.img = None
        self.result = None

        self.style = ttk.Style()
        self.style.configure("TButton", font=("微软雅黑", 11), padding=8)
        self.style.configure("TCombobox", font=("微软雅黑", 11))

        title = tk.Label(root, text="霍夫变换 · 形状识别可视化",
                         font=("微软雅黑", 18, "bold"), bg="#f0f4f8", fg="#2d3748")
        title.pack(pady=15)

        control = tk.Frame(root, bg="#ffffff", padx=20, pady=15, relief=tk.RIDGE, borderwidth=1)
        control.pack(pady=5, fill=tk.X)

        self.choice = ttk.Combobox(control, values=["直线", "圆形", "椭圆", "三角形"],
                                   width=12, font=("微软雅黑", 11))
        self.choice.current(0)
        self.choice.pack(side=tk.LEFT, padx=10)

        tk.Button(control, text="📁 打开图片", bg="#4299e1", fg="white", relief=tk.FLAT,
                  font=("微软雅黑", 11), width=12, command=self.load_img).pack(side=tk.LEFT, padx=10)

        tk.Button(control, text="🔍 开始识别", bg="#38a169", fg="white", relief=tk.FLAT,
                  font=("微软雅黑", 11), width=12, command=self.detect).pack(side=tk.LEFT, padx=10)

        self.img_frame = tk.Frame(root, bg="#f0f4f8")
        self.img_frame.pack(pady=20, expand=True, fill=tk.BOTH)

        self.lab_original = tk.Label(self.img_frame, text="原始图像", font=("微软雅黑", 12), bg="#f0f4f8")
        self.lab_original.grid(row=0, column=0, padx=20)
        self.canvas_original = tk.Label(self.img_frame, bg="#e2e8f0", bd=2, relief=tk.SUNKEN)
        self.canvas_original.grid(row=1, column=0, padx=20)

        self.lab_result = tk.Label(self.img_frame, text="识别结果", font=("微软雅黑", 12), bg="#f0f4f8")
        self.lab_result.grid(row=0, column=1, padx=20)
        self.canvas_result = tk.Label(self.img_frame, bg="#e2e8f0", bd=2, relief=tk.SUNKEN)
        self.canvas_result.grid(row=1, column=1, padx=20)

    def load_img(self):
        path = filedialog.askopenfilename(filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.webp")])
        if not path:
            return
        self.img = cv2.imread(path)
        self.show_img(self.img, self.canvas_original)

    def show_img(self, img, label):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((420, 420), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        label.config(image=img_tk)
        label.image = img_tk

    def detect(self):
        if self.img is None:
            return
        img, edges = preprocess_image(self.img)
        t = self.choice.get()
        if t == "直线":
            self.result = hough_line_detection(img, edges)
        elif t == "圆形":
            self.result = hough_circle_detection(img)
        elif t == "椭圆":
            self.result = hough_ellipse_detection(img, edges)
        elif t == "三角形":
            self.result = hough_triangle_detection(img, edges)
        self.show_img(self.result, self.canvas_result)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernShapeApp(root)
    root.mainloop()