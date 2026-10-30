for run in {86..88}; do
  make RUN_ID="$run" all || break;
done
