
PERIOD=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
QUOTA=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)

if [ "$QUOTA" -gt "-1" ]; then
    # -1 表示没有设置限制
    EFFECTIVE_CORES=$(echo "scale=2; $QUOTA / $PERIOD" | bc)
    echo "容器受限核心数: $EFFECTIVE_CORES"
else
    # 如果没有配额限制，则使用宿主机的逻辑核心数
    echo "容器没有 CPU 配额限制，逻辑核心数: $(nproc)"
fi
