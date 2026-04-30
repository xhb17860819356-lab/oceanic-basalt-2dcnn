import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ==============================
# Matplotlib 投稿级绘图设置
# ==============================

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import shap
from tensorflow.keras import layers, models, callbacks, optimizers


# ==============================
# 统一导出图件函数
# ==============================

def save_figure(fig_name):
    plt.savefig(f"{fig_name}.pdf")
    plt.savefig(f"{fig_name}.eps")
    plt.savefig(f"{fig_name}.png", dpi=600)
# ==============================
# 0. 固定随机种子
# ==============================
np.random.seed(42)
tf.random.set_seed(42)

# ==============================
# 1. 读取数据
# ==============================
file_path = "./data/combined_for_DL_13000.xlsx"
df = pd.read_excel(file_path)

elements = [
    'SiO2_wt', 'TiO2_wt', 'Al2O3_wt', 'FeO_wt', 'MgO_wt', 'CaO_wt',
    'Na2O_wt', 'K2O_wt', 'P2O5_wt',
    'Nb_ppm', 'Zr_ppm', 'Y_ppm', 'Th_ppm', 'Yb_ppm',
    'Nb_Yb', 'Th_Yb', 'log_Nb_Yb', 'log_Th_Yb'
]

label_col = 'label'

df = df.dropna(subset=elements + [label_col]).copy()

X = df[elements].copy()
y = df[label_col].copy()

# ==============================
# 图A. 相关性热图
# ==============================
corr = X.corr()


rename_dict = {
    'SiO2_wt': 'SiO$_2$',
    'TiO2_wt': 'TiO$_2$',
    'Al2O3_wt': 'Al$_2$O$_3$',
    'FeO_wt': 'FeO',
    'MgO_wt': 'MgO',
    'CaO_wt': 'CaO',
    'Na2O_wt': 'Na$_2$O',
    'K2O_wt': 'K$_2$O',
    'P2O5_wt': 'P$_2$O$_5$',
    'Nb_ppm': 'Nb',
    'Zr_ppm': 'Zr',
    'Y_ppm': 'Y',
    'Th_ppm': 'Th',
    'Yb_ppm': 'Yb',
    'Nb_Yb': 'Nb/Yb',
    'Th_Yb': 'Th/Yb',
    'log_Nb_Yb': 'log(Nb/Yb)',
    'log_Th_Yb': 'log(Th/Yb)'
}

corr = corr.rename(index=rename_dict, columns=rename_dict)

# 只显示下三角，避免重复信息
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    corr,
    mask=mask,
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={
        'shrink': 0.85,
        'pad': 0.02,
        'aspect': 30,
        'label': 'Pearson correlation'
    }
)

plt.title("Correlation matrix of geochemical features", fontsize=13, pad=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)


for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)

plt.tight_layout()
save_figure("Fig_correlation_heatmap_pub")
plt.close()

# ==============================
# 2. 数据预处理
# ==============================
raw_log_cols = [
    'SiO2_wt', 'TiO2_wt', 'Al2O3_wt', 'FeO_wt', 'MgO_wt', 'CaO_wt',
    'Na2O_wt', 'K2O_wt', 'P2O5_wt',
    'Nb_ppm', 'Zr_ppm', 'Y_ppm', 'Th_ppm', 'Yb_ppm',
    'Nb_Yb', 'Th_Yb'
]

X[raw_log_cols] = np.log10(X[raw_log_cols] + 1e-6)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ==============================
# 3. 3×6 满矩阵编码
# ==============================
layout_3x6 = [
    ['SiO2_wt', 'TiO2_wt', 'Al2O3_wt', 'FeO_wt', 'MgO_wt', 'CaO_wt'],
    ['Na2O_wt', 'K2O_wt', 'P2O5_wt', 'Nb_ppm', 'Zr_ppm', 'Y_ppm'],
    ['Th_ppm', 'Yb_ppm', 'Nb_Yb', 'Th_Yb', 'log_Nb_Yb', 'log_Th_Yb']
]

feature_index = {name: i for i, name in enumerate(elements)}

def to_matrix_3x6(X_array, layout, feature_map):
    n = X_array.shape[0]
    out = np.zeros((n, 3, 6, 1), dtype=np.float32)
    for i in range(3):
        for j in range(6):
            feat = layout[i][j]
            out[:, i, j, 0] = X_array[:, feature_map[feat]]
    return out

X_train_img = to_matrix_3x6(X_train, layout_3x6, feature_index)
X_test_img = to_matrix_3x6(X_test, layout_3x6, feature_index)


# ==============================
# 图B. 3×6 输入矩阵布局示意图
# ==============================


layout_display = [
    ['SiO$_2$', 'TiO$_2$', 'Al$_2$O$_3$', 'FeO', 'MgO', 'CaO'],
    ['Na$_2$O', 'K$_2$O', 'P$_2$O$_5$', 'Nb', 'Zr', 'Y'],
    ['Th', 'Yb', 'Nb/Yb', 'Th/Yb', 'log(Nb/Yb)', 'log(Th/Yb)']
]

# 定义三种类别的颜色：主量元素、微量元素、元素比值

category_colors = ['#DCEAF7', '#E8F3E6', '#FCE8D8']  

fig, ax = plt.subplots(figsize=(11, 4.5))

for i in range(3):
    for j in range(6):
        # 计算当前格子的一维顺序索引 (0 到 17)
        idx = i * 6 + j
        
        # 根据特征的顺序分配颜色
        if idx < 9:
            # 前9个: 主量元素 (SiO2 ~ P2O5)
            bg_color = category_colors[0]
        elif idx < 14:
            # 中间5个: 微量元素 (Nb ~ Yb)
            bg_color = category_colors[1]
        else:
            # 最后4个: 元素比值 (Nb/Yb ~ log(Th/Yb))
            bg_color = category_colors[2]

        rect = plt.Rectangle(
            (j, i), 1, 1,
            facecolor=bg_color,  # 应用根据类别判断出的颜色
            edgecolor='black',
            linewidth=1.0
        )
        ax.add_patch(rect)
        ax.text(
            j + 0.5, i + 0.5, layout_display[i][j],
            ha='center', va='center',
            fontsize=11
        )

# 设置坐标轴范围
ax.set_xlim(0, 6)
ax.set_ylim(3, 0)
ax.set_xticks([])
ax.set_yticks([])

# 添加标题
ax.set_title("Structured 3×6 layout of geochemical features for CNN input", fontsize=13, pad=10)

# 去掉边框外框
for spine in ax.spines.values():
    spine.set_visible(False)

# 保存图像
plt.tight_layout()
save_figure("Fig_input_layout_3x6_by_category")  # 调用你的自定义保存函数
plt.close()
# ==============================
# 4. 定义 3 种不同 CNN（异构集成）
# ==============================
from tensorflow.keras import regularizers

def build_cnn_a(input_shape, n_classes, lr=2e-4):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(64, (2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Conv2D(64, (2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Dropout(0.25),

        layers.Conv2D(128, (2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.35),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_cnn_b(input_shape, n_classes, lr=1.5e-4):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Conv2D(64, (2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Dropout(0.35),

        layers.Conv2D(128, (2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.GlobalAveragePooling2D(),

        layers.Dense(96, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.45),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_cnn_c(input_shape, n_classes, lr=2e-4):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (1, 1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, (2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(128, (2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dropout(0.30)(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ==============================
# 5. 训练函数
# ==============================
def train_one_model(model_builder, seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = model_builder(
        input_shape=(3, 6, 1),
        n_classes=len(np.unique(y_encoded))
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=0
    )

    history = model.fit(
        X_train_img, y_train,
        epochs=220,
        batch_size=24,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    y_prob = model.predict(X_test_img, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    test_acc = accuracy_score(y_test, y_pred)

    best_val_acc = max(history.history['val_accuracy'])

    return model, history, y_prob, y_pred, test_acc, best_val_acc


# ==============================
# 6. 单模型结果（取 CNN-A + seed=42 作为 single CNN）
# ==============================
model, history, y_prob, y_pred, cnn_acc, best_val_acc = train_one_model(build_cnn_a, 42)

print("Single CNN Accuracy:", cnn_acc)
print(classification_report(y_test, y_pred))


# ==============================
# 7. 异构 + 多 seed 加权集成
# ==============================
ensemble_specs = [
    ("A", build_cnn_a, 7),
    ("A", build_cnn_a, 42),
    ("B", build_cnn_b, 21),
    ("B", build_cnn_b, 99),
    ("C", build_cnn_c, 77),
    ("C", build_cnn_c, 123),
]

all_probs = []
all_weights = []
all_accs = []

for name, builder, seed in ensemble_specs:
    _, _, prob_s, pred_s, acc_s, val_acc_s = train_one_model(builder, seed)
    all_probs.append(prob_s)
    all_weights.append(val_acc_s)
    all_accs.append(acc_s)
    print(f"Model {name}, seed {seed}: test_acc={acc_s:.4f}, best_val_acc={val_acc_s:.4f}")

all_weights = np.array(all_weights, dtype=np.float64)
all_weights = all_weights / all_weights.sum()

avg_prob = np.zeros_like(all_probs[0])
for w, p in zip(all_weights, all_probs):
    avg_prob += w * p

ensemble_pred = np.argmax(avg_prob, axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print("Ensemble CNN Accuracy:", ensemble_acc)
print(classification_report(y_test, ensemble_pred))


# ==============================
# 8. RF 对比
# ==============================
rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("RF Accuracy:", rf_acc)

# ==============================
# RF 特征重要性
# ==============================

importances = rf.feature_importances_

plt.figure(figsize=(8,6))
sns.barplot(x=importances, y=elements)
plt.title("Feature Importance (RF)")
plt.xlabel("Importance")
plt.ylabel("Geochemical Features")
plt.tight_layout()
save_figure("Fig_feature_importance")
plt.close()

# ==============================
# 8. 图件：训练曲线（单个CNN）
# ==============================
plt.figure(figsize=(7, 5))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Curve (Single CNN)")
plt.legend()
plt.tight_layout()
save_figure("Fig_training_curve_cnn3x6.png")
plt.close()

# ==============================
# 图D. 计数混淆矩阵
# ==============================
cm = confusion_matrix(y_test, ensemble_pred)


class_names = le.inverse_transform(np.unique(y_encoded))

plt.figure(figsize=(5.8, 5.0))
ax = sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=True,
    square=True,
    linewidths=0.6,
    linecolor='white',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={
        'shrink': 0.85,
        'pad': 0.02,
        'aspect': 25,
        'label': 'Count'
    }
)

ax.set_xlabel("Predicted class", fontsize=11)
ax.set_ylabel("True class", fontsize=11)
ax.tick_params(axis='x', rotation=0, labelsize=10)
ax.tick_params(axis='y', rotation=0, labelsize=10)


ax.set_title("Confusion matrix of Ensemble CNN", fontsize=12, pad=10)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)

plt.tight_layout()
save_figure("Fig_confusion_matrix_ensemble")
plt.close()

# ==============================
# 图E. 归一化混淆矩阵
# ==============================
cm_norm = confusion_matrix(y_test, ensemble_pred, normalize='true')

plt.figure(figsize=(5.8, 5.0))
ax = sns.heatmap(
    cm_norm,
    annot=True,
    fmt='.2f',
    cmap='Blues',
    vmin=0,
    vmax=1,
    cbar=True,
    square=True,
    linewidths=0.6,
    linecolor='white',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={
        'shrink': 0.85,
        'pad': 0.02,
        'aspect': 25,
        'label': 'Proportion'
    }
)

ax.set_xlabel("Predicted class", fontsize=11)
ax.set_ylabel("True class", fontsize=11)
ax.tick_params(axis='x', rotation=0, labelsize=10)
ax.tick_params(axis='y', rotation=0, labelsize=10)
ax.set_title("Normalized confusion matrix of Ensemble CNN", fontsize=12, pad=10)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)

plt.tight_layout()
save_figure("Fig_confusion_matrix_norm")
plt.close()

# ==============================
# 图F. 各类别 Precision / Recall / F1-score 对比图
# ==============================
report_dict = classification_report(
    y_test,
    ensemble_pred,
    target_names=class_names,
    output_dict=True
)

metric_df = pd.DataFrame({
    'Class': class_names,
    'Precision': [report_dict[c]['precision'] for c in class_names],
    'Recall': [report_dict[c]['recall'] for c in class_names],
    'F1-score': [report_dict[c]['f1-score'] for c in class_names]
})

metric_long = metric_df.melt(
    id_vars='Class',
    var_name='Metric',
    value_name='Score'
)

# 更适合论文的柔和配色
palette = {
    'Precision': '#4C78A8',  # 柔和蓝
    'Recall': '#F58518',     # 柔和橙
    'F1-score': '#54A24B'    # 柔和绿
}

fig, ax = plt.subplots(figsize=(7.4, 4.8))

sns.barplot(
    data=metric_long,
    x='Class',
    y='Score',
    hue='Metric',
    palette=palette,
    edgecolor='black',
    linewidth=0.8,
    ax=ax
)


ax.set_ylim(0, 1.02)
ax.set_xlabel("")
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Class-wise evaluation metrics of Ensemble CNN", fontsize=13, pad=8)


ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)

# 坐标轴与刻度
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=11)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)


legend = ax.legend(
    title=None,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.00),
    ncol=3,
    frameon=True,
    fontsize=10
)
legend.get_frame().set_linewidth(0.8)

# 数值标注
for container in ax.containers:
    labels = [f"{bar.get_height():.2f}" for bar in container]
    ax.bar_label(container, labels=labels, padding=2, fontsize=9)

plt.tight_layout()
save_figure("Fig_class_metrics_ensemble")
plt.close()

# ==============================
# 10. t-SNE
# ==============================
feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[-3].output
)

features = feature_extractor.predict(X_test_img, verbose=0)

# 为避免图过密，随机抽样展示
n_show = min(1000, features.shape[0])
idx = np.random.RandomState(42).choice(features.shape[0], n_show, replace=False)

tsne = TSNE(
    n_components=2,
    random_state=42,
    init="pca",
    learning_rate="auto"
)
X_tsne = tsne.fit_transform(features[idx])

# 取抽样样本对应的真实类别名称
label_names = le.inverse_transform(y_test[idx])

# 固定颜色
color_map = {
    'IAB': '#E6C229',   # 金黄
    'MORB': '#2A9D8F',  # 青绿
    'OIB': '#3B0F70'    # 深紫
}

fig, ax = plt.subplots(figsize=(6.4, 5.2))

for cls in ['IAB', 'MORB', 'OIB']:
    mask = (label_names == cls)
    ax.scatter(
        X_tsne[mask, 0],
        X_tsne[mask, 1],
        s=28,
        c=color_map[cls],
        label=cls,
        alpha=0.95,
        edgecolors='none'
    )

ax.set_xlabel("t-SNE dimension 1", fontsize=11)
ax.set_ylabel("t-SNE dimension 2", fontsize=11)
ax.set_title("t-SNE visualization of single CNN features", fontsize=13, pad=10)

# 图例
legend = ax.legend(
    title=None,
    loc='best',
    frameon=True,
    fontsize=10
)
legend.get_frame().set_linewidth(0.8)

# 坐标轴样式
ax.tick_params(axis='both', labelsize=10)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.tight_layout()
save_figure("Fig_tsne")
plt.close()

# ==============================
# 图C. 模型准确率对比图
# ==============================

model_names = ['Single CNN', 'Ensemble CNN', 'RF']
model_scores = [cnn_acc, ensemble_acc, rf_acc]


bar_colors = ['#A6CEE3', '#F4A6A6', '#B2DF8A']

fig, ax = plt.subplots(figsize=(6.8, 4.8))

bars = ax.bar(
    model_names,
    model_scores,
    width=0.45,
    color=bar_colors,
    edgecolor='black',
    linewidth=1.2,
    zorder=3
)

# 数值标注
for bar, score in zip(bars, model_scores):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        score + 0.0015,
        f"{score:.3f}",
        ha='center',
        va='bottom',
        fontsize=10
    )

ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Performance comparison of different models", fontsize=14, pad=10)


ymin = min(model_scores) - 0.01
ymax = max(model_scores) + 0.015
ax.set_ylim(ymin, ymax)


ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5)
ax.set_axisbelow(True)

# 统一字体大小
ax.tick_params(axis='x', labelsize=11)
ax.tick_params(axis='y', labelsize=11)

plt.tight_layout()
save_figure("Fig_model_comparison")
plt.close()

# ==============================
# 11. SHAP 可解释性分析（基于代表性 single CNN）
# ==============================
# 说明：
# 由于最终最优模型是异构集成 Ensemble CNN，直接对整个集成模型做 SHAP 比较复杂，
# 因此这里采用性能较稳定、结构清晰的代表性 single CNN（变量名 model）进行解释。
# 该 model 已在前文中训练完成，可直接用于可解释性分析。

# ---------- 11.1 准备背景集与解释样本 ----------
rng = np.random.RandomState(42)


bg_size = min(200, X_train_img.shape[0])
explain_size = min(500, X_test_img.shape[0])

bg_idx = rng.choice(X_train_img.shape[0], bg_size, replace=False)
explain_idx = rng.choice(X_test_img.shape[0], explain_size, replace=False)

X_bg = X_train_img[bg_idx]
X_explain_img = X_test_img[explain_idx]
X_explain_tab = X_test[explain_idx]   # 用于绘图显示（18个特征的一维形式）
y_explain = y_test[explain_idx]


pred_prob_explain = model.predict(X_explain_img, verbose=0)
pred_class_explain = np.argmax(pred_prob_explain, axis=1)

# ---------- 11.2 计算 SHAP 值 ----------
# 对 CNN 模型优先采用 GradientExplainer
explainer = shap.GradientExplainer(model, X_bg)
raw_shap_values = explainer.shap_values(X_explain_img)

# ---------- 11.3 统一整理 SHAP 输出格式 ----------
# 目标格式：shap_3d.shape = (n_samples, n_features, n_classes)
def format_shap_values(raw_shap, n_classes):
    if isinstance(raw_shap, list):

        formatted = np.stack(
            [np.array(v).reshape(np.array(v).shape[0], -1) for v in raw_shap],
            axis=-1
        )
        return formatted

    arr = np.array(raw_shap)


    if arr.ndim == 5 and arr.shape[-1] == n_classes:
        return arr.reshape(arr.shape[0], -1, n_classes)

    if arr.ndim == 5 and arr.shape[0] == n_classes:
        arr = np.moveaxis(arr, 0, -1)   # -> (n_samples, 3, 6, 1, n_classes)
        return arr.reshape(arr.shape[0], -1, n_classes)

    if arr.ndim == 3 and arr.shape[-1] == n_classes:
        return arr

    raise ValueError(f"无法识别 SHAP 输出形状: {arr.shape}")

n_classes = len(class_names)
shap_3d = format_shap_values(raw_shap_values, n_classes)

# ---------- 11.4 图1：全局 SHAP summary beeswarm ----------


shap_pred_class = np.zeros((X_explain_img.shape[0], len(elements)))
for i in range(X_explain_img.shape[0]):
    shap_pred_class[i, :] = shap_3d[i, :, pred_class_explain[i]]


feature_names_pub = [
    'SiO$_2$', 'TiO$_2$', 'Al$_2$O$_3$', 'FeO', 'MgO', 'CaO',
    'Na$_2$O', 'K$_2$O', 'P$_2$O$_5$',
    'Nb', 'Zr', 'Y', 'Th', 'Yb',
    'Nb/Yb', 'Th/Yb', 'log(Nb/Yb)', 'log(Th/Yb)'
]

plt.figure(figsize=(8.2, 6.2))
shap.summary_plot(
    shap_pred_class,
    features=X_explain_tab,
    feature_names=feature_names_pub,
    show=False,
    max_display=18
)
plt.title("Global SHAP summary of the representative CNN model", fontsize=13, pad=10)
plt.tight_layout()
save_figure("Fig_shap_summary_beeswarm")
plt.close()

# ---------- 11.5 图2：三分类别的 class-wise SHAP bar plot ----------


fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharex=False)

bar_colors = {
    'IAB': '#E6C229',   # 金黄
    'MORB': '#2A9D8F',  # 青绿
    'OIB': '#3B0F70'    # 深紫
}

for class_idx, class_name in enumerate(class_names):
    ax = axes[class_idx]

    # 只取真实属于该类的样本
    mask = (y_explain == class_idx)

    if mask.sum() == 0:
        ax.set_title(class_name)
        ax.text(0.5, 0.5, 'No samples', ha='center', va='center', transform=ax.transAxes)
        continue

    # 取该类输出对应的 SHAP 值
    shap_class = shap_3d[mask, :, class_idx]

    # 平均绝对 SHAP 值
    mean_abs_shap = np.mean(np.abs(shap_class), axis=0)

    # 取前10个最重要特征
    top_idx = np.argsort(mean_abs_shap)[-10:][::-1]
    top_features = [feature_names_pub[i] for i in top_idx]
    top_values = mean_abs_shap[top_idx]

    sns.barplot(
        x=top_values,
        y=top_features,
        ax=ax,
        color=bar_colors[class_name],
        edgecolor='black',
        linewidth=0.8
    )

    ax.set_title(class_name, fontsize=12, pad=8)
    ax.set_xlabel("mean(|SHAP value|)", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

plt.suptitle("Class-wise SHAP importance of geochemical features", fontsize=13, y=1.02)
plt.tight_layout()
save_figure("Fig_shap_classwise_bar")
plt.close()

print("全部图件已生成。")