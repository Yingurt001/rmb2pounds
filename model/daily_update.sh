#!/bin/bash
# 每日重训预测并推送到 GitHub (本地 launchd 调用)
REPO="/Users/zhangying/Documents/My Project/rmb2pounds"
cd "$REPO" || exit 1
echo "===== $(date) ====="
/opt/anaconda3/bin/python3 model/forecast.py 2>&1 | grep -vi "warn\|numexpr\|bottleneck\|pyarrow\|from pandas"
git add predictions.json backtest_report.md
if git diff --cached --quiet; then
  echo "无变化，跳过提交"
else
  git commit -q -m "chore: 每日预测更新 $(date -u +%F)" && git push -q && echo "已推送"
fi
