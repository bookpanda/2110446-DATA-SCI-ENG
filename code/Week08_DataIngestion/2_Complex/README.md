# Strcture
## Producer A, B
- produce signals

## file writer consumer
- write signals to log every signal

## counter service
- checks every 10 seconds to see how many signals occured
- publishes to notifcation topic, reset the counters

## notifier consumer
- checks every nofication messages
