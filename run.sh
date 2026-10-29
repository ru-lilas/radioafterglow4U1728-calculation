for run in {80..85}; do
  make RUN_ID="$run" all || break;
done
