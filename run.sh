for run in {60..76}; do
  make RUN_ID="$run" all || break;
done
