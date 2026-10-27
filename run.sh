for run in {50..56}; do
  make RUN_ID="$run" all || break;
done
