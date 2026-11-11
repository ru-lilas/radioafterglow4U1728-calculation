echo "start: $1"
echo "stop: $2"

for ((run=$1; run<=$2; run++)); do
  make RUN_ID="$run" all || break
done
