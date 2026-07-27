# DeepSeek V3 32 层 proxy 在 Ironwood 上的预训练（64 芯片 / XPK）

本配方在 64 芯片（4x4x4）的 Ironwood GKE 集群上，通过
[XPK](https://github.com/AI-Hypercomputer/xpk) 运行 DeepSeek V3 的
**32 层缩减版**（约 221B 参数）预训练，使用 MaxText main 分支。

需要交付给客户、希望对方能直接看懂部署内容时，用 [../k8s](../k8s/README.md)
的原生 manifest 版本。

> English version: [README.en.md](README.en.md)

## 模型配置的依据

减层的原因、proxy 模型「只减深度不动宽度」的原则、以及为什么
`num_experts` 必须保持 256，见
[k8s 版 README](../k8s/README.md#为什么要减层)。这里只列最终参数。

```
base_num_decoder_layers=32     # 完整模型 61 层
num_experts=256                # 不动，见约束说明
mtp_num_layers=1
mtp_loss_scaling_factor=0.1
per_device_batch_size=4.0      # 64 芯片下每卡权重分片是 128 芯片的两倍
```

其余参数（并行策略、XLA flags、attention 配置）与
[4k-bf16-tpu7x-4x4x8-latest](../../4k-bf16-tpu7x-4x4x8-latest/xpk/README.md)
完全一致。

## 任务配置

-   序列长度：4096
-   精度：bf16
-   芯片数：64（4x4x4 拓扑，16 个 VM × 4 芯片）
-   模型：DeepSeek V3 32 层 proxy，约 221B 参数
-   数据集：synthetic（无外部依赖）
-   Checkpoint：关闭

## 前置条件与环境安装

XPK 安装、Docker 配置、镜像构建步骤与
[4k-bf16-tpu7x-4x4x8-latest](../../4k-bf16-tpu7x-4x4x8-latest/xpk/README.md)
相同。注意镜像必须是 **runner** 镜像，base 镜像不含 MaxText 源码。

## 创建集群

TPU v7 不支持自动创建 placement policy，必须先建 4x4x4 的：

```bash
gcloud compute resource-policies create workload-policy tpu7x-64chip \
  --region=${REGION} --project=${PROJECT_ID} \
  --type=HIGH_THROUGHPUT --accelerator-topology=4x4x4
```

```bash
xpk cluster create \
  --cluster=${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --tpu-type=tpu7x-4x4x4 \
  --num-slices=1 \
  --reservation=${RESERVATION_NAME}
```

容量紧张时，固定节点数比 autoscaling 更容易拿到卡，原因见
[k8s 版说明](../k8s/README.md#常驻-vs-弹性在容量紧张时优先常驻)。

## 运行

```bash
cd ~
git clone https://github.com/ai-hypercomputer/tpu-recipes.git
cd tpu-recipes/training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x4x4-32l/xpk

# 编辑 run_recipe.sh 顶部的 export 变量
nano ./run_recipe.sh

chmod +x run_recipe.sh
./run_recipe.sh
```

## 监控

```bash
xpk workload list --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}

kubectl get pods | grep ${WORKLOAD_NAME}
kubectl logs <POD_NAME> | grep 'completed step'
kubectl logs <POD_NAME> | grep RESOURCE_EXHAUSTED
```

## 读取结果

**本配方是 32 层 proxy，其性能数字不能直接与 61 层完整模型对比。**

指标定义与单位换算（`TFLOP/s/device` 是 per TensorCore，per-chip 要 ×2）见
[k8s 版说明](../k8s/README.md#读取结果)。

## 验证状态

<!-- BENCHMARK-PLACEHOLDER -->
待采集。已验证部分见 [k8s 版](../k8s/README.md#验证状态)。

## 清理

```bash
xpk workload delete --workload ${WORKLOAD_NAME} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}
xpk cluster delete --cluster ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT_ID}
```
