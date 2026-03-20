# AskSpark Test Report
Generated: 2026-03-20 08:19:29

## Summary
- Overall Status: FAILED
- Total Duration: 181.54s

## Test Suite Results
### Unit Tests
- Status: FAIL
- Duration: 8.15s

### Integration Tests
- Status: FAIL
- Duration: 48.20s
- Error: --- Logging error ---
Traceback (most recent call last):
  File "/opt/homebrew/Caskroom/miniforge/base/lib/python3.12/logging/__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
ValueError: I/O operation on closed file.
Call stack:
  File "/Users/startferanmi/Documents/AskSpark/ai_consultant_env/lib/python3.12/site-packages/huggingface_hub/utils/_http.py", line 383, in close_session
    client.close()
  File "/Users/startferanmi/Documents/AskSpark/ai_consultant_env/lib/python3.12/site-packages/httpx/_client.py", line 1270, in close
    self._transport.close()
  File "/Users/startferanmi/Documents/AskSpark/ai_consultant_env/lib/python3.12/site-packages/httpx/_transports/default.py", line 262, in close
    self._pool.close()
  File "/Users/startferanmi/Documents/AskSpark/ai_consultant_env/lib/python3.12/site-packages/httpcore/_sync/connection_pool.py", line 353, in close
    self._close_connections(closing_connections)
  File "/Users/startferanmi/Documents/AskSpark/ai_consultant_env/lib/python3.12/site-packages/httpcore/_sync/connection_pool.py", line 345, in _close_connections
    connection.close()
  File "/Users/startferanmi/Documents/AskSpark/ai_consultant_env/lib/python3.12/site-packages/httpcore/_sync/connection.py", line 172, in close
    with Trace("close", logger, None, {}):
  File "/Users/startferanmi/Documents/AskSpark/ai_consultant_env/lib/python3.12/site-packages/httpcore/_trace.py", line 52, in __enter__
    self.trace(f"{self.name}.started", info)
  File "/Users/startferanmi/Documents/AskSpark/ai_consultant_env/lib/python3.12/site-packages/httpcore/_trace.py", line 47, in trace
    self.logger.debug(message)
Message: 'close.started'
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "/opt/homebrew/Caskroom/miniforge/base/lib/python3.12/logging/__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
ValueError: I/O operation on closed file.
Call stack:
  File "/Users/startferanmi/Documents/AskSpark/ai_consultant_env/lib/python3.12/site-packages/huggingface_hub/utils/_http.py", line 383, in close_session
    client.close()
  File "/Users/startferanmi/Documents/AskSpark/ai_consultant_env/lib/python3.12/site-packages/httpx/_client.py", line 1270, in close
    self._transport.close()
  File "/Users/startferanmi/Documents/AskSpark/ai_consultant_env/lib/python3.12/site-packages/httpx/_transports/default.py", line 262, in close
    self._pool.close()
  File "/Users/startferanmi/Documents/AskSpark/ai_consultant_env/lib/python3.12/site-packages/httpcore/_sync/connection_pool.py", line 353, in close
    self._close_connections(closing_connections)
  File "/Users/startferanmi/Documents/AskSpark/ai_consultant_env/lib/python3.12/site-packages/httpcore/_sync/connection_pool.py", line 345, in _close_connections
    connection.close()
  File "/Users/startferanmi/Documents/AskSpark/ai_consultant_env/lib/python3.12/site-packages/httpcore/_sync/connection.py", line 172, in close
    with Trace("close", logger, None, {}):
  File "/Users/startferanmi/Documents/AskSpark/ai_consultant_env/lib/python3.12/site-packages/httpcore/_trace.py", line 64, in __exit__
    self.trace(f"{self.name}.complete", info)
  File "/Users/startferanmi/Documents/AskSpark/ai_consultant_env/lib/python3.12/site-packages/httpcore/_trace.py", line 47, in trace
    self.logger.debug(message)
Message: 'close.complete'
Arguments: ()


### Performance Tests
- Status: FAIL
- Duration: 33.88s

### Data Flow Validation
- Status: FAIL
- Duration: 6.98s

## Recommendations
- Fix failing test suites before deployment
- Review error messages for troubleshooting
- Monitor performance metrics in production
- Regularly run data flow validation