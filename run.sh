for run in {74..77}; do
  make RUN_ID="$run" all || break;
done
