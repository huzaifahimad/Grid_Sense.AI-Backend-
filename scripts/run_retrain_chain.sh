cd "$(dirname "$0")/.." || exit 9
: > _audit_out/chain.status
run () {
  echo "[chain] start $* $(date -u +%H:%M:%S)Z" >> _audit_out/chain.stdout.log
  python -u -m "$1" $2 > "_audit_out/${1##*.}.log" 2>&1
  code=$?
  echo "$1 exit=$code $(date -u +%H:%M:%S)Z" >> _audit_out/chain.status
  echo "[chain] done $1 exit=$code $(date -u +%H:%M:%S)Z" >> _audit_out/chain.stdout.log
}
run models.train_baselines
run models.risk_classifier
run models.calibrate_forecast_intervals
run models.closed_loop_validation --strict
echo "CHAIN_FINISHED $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> _audit_out/chain.status
