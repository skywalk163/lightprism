#!/bin/sh
# 把 main 推到三个远端并逐个复核。
# 用法：sh scripts/push-all.sh [branch]
# 复核方式按 lightharness 的教训：**不信 `git push | tail`**，逐远端 ls-remote 比对 SHA。
set -eu
BRANCH="${1:-main}"
LOCAL=$(git rev-parse "$BRANCH")

for remote in origin github gitcode; do
    echo "== push $remote ($BRANCH)"
    git push "$remote" "$BRANCH"
done

echo
echo "== 复核（local $LOCAL）"
fail=0
for remote in origin github gitcode; do
    remote_sha=$(git ls-remote "$remote" "refs/heads/$BRANCH" | awk '{print $1}')
    if [ "$remote_sha" = "$LOCAL" ]; then
        echo "  ✅ $remote $remote_sha"
    else
        echo "  ❌ $remote ${remote_sha:-<missing>}"
        fail=1
    fi
done
exit "$fail"
