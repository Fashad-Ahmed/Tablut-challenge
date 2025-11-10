# Submission Checklist

Use this checklist to ensure your agent is ready for competition submission.

## Pre-Submission Testing

### Local Testing
- [ ] Agent runs successfully on macOS M1 Pro
- [ ] Agent connects to Java server
- [ ] Agent can play as WHITE
- [ ] Agent can play as BLACK
- [ ] Agent completes full games without errors
- [ ] Agent respects 60-second time limit
- [ ] Agent handles illegal moves gracefully
- [ ] Agent handles network errors gracefully

### VM Testing
- [ ] Agent runs in Linux Debian 64-bit VM
- [ ] All dependencies install correctly
- [ ] Model loads successfully (if using)
- [ ] Agent plays games in VM environment
- [ ] Performance acceptable in VM (4 CPUs, 8GB RAM)

### Functionality
- [ ] Minimax search works correctly
- [ ] Alpha-beta pruning functioning
- [ ] Transposition table working
- [ ] Move ordering improves performance
- [ ] Value network evaluation (if using model)
- [ ] Heuristic fallback works (if model unavailable)
- [ ] Legal move generation correct
- [ ] Terminal state detection correct

## Code Quality

### Formatting
- [ ] Code formatted with black
- [ ] Imports sorted with isort
- [ ] No linting errors (ruff)
- [ ] Type hints added where appropriate

### Testing
- [ ] All unit tests pass
- [ ] Test coverage acceptable (>70%)
- [ ] Integration tests pass
- [ ] Performance tests pass

### Documentation
- [ ] README.md complete and accurate
- [ ] ARCHITECTURE.md describes system
- [ ] API.md documents protocol
- [ ] Code has docstrings
- [ ] TODO markers documented

## Submission Files

### Virtual Machine
- [ ] VM image created (VirtualBox .ova or .vdi)
- [ ] VM contains all dependencies
- [ ] VM contains agent code
- [ ] VM contains trained model (if using)
- [ ] VM size < 30GB
- [ ] VM tested and working

### Submission .txt File
- [ ] OneDrive link to VM included
- [ ] Execution command documented
- [ ] Optional parameters documented
- [ ] File uploaded to VIRTUALE

### Code Repository (if using GitHub)
- [ ] Repository is public
- [ ] README.md is clear
- [ ] Code is well-organized
- [ ] License included

## Performance Verification

### Time Management
- [ ] Moves computed within 60 seconds
- [ ] Time buffer (5s) reserved for transmission
- [ ] Iterative deepening working
- [ ] Early termination on winning moves

### Memory Management
- [ ] Transposition table size limited
- [ ] No memory leaks
- [ ] Disk usage reasonable (< 1GB logs)
- [ ] No excessive output

### Model Performance (if using)
- [ ] Model inference < 100ms
- [ ] Model size reasonable (< 100MB)
- [ ] Model loads quickly
- [ ] Model works on CPU

## Competition Requirements

### Command-Line Interface
- [ ] Agent accepts required arguments:
  - Player role (WHITE/BLACK)
  - Timeout (seconds)
  - Server IP
- [ ] Agent accepts optional arguments:
  - Model path
  - Search depth
  - Random seed
  - Log level

### Communication Protocol
- [ ] TCP socket connection working
- [ ] JSON message format correct
- [ ] Length-prefixed protocol implemented
- [ ] Player name sent correctly
- [ ] Moves sent in correct format
- [ ] State received and parsed correctly

### Game Rules
- [ ] Legal moves generated correctly
- [ ] Illegal moves rejected
- [ ] Terminal states detected
- [ ] Win/loss/draw conditions handled
- [ ] Special rules implemented (Castle, Camps)

## Final Checks

### Before Submission
- [ ] All tests pass
- [ ] Code reviewed
- [ ] Documentation complete
- [ ] VM tested thoroughly
- [ ] Submission .txt file ready
- [ ] OneDrive link working
- [ ] Execution command tested

### Submission Day
- [ ] VM image uploaded to OneDrive
- [ ] .txt file submitted to VIRTUALE
- [ ] Submission before deadline (November 23, 23:59 Italian time)
- [ ] Backup copy saved

## Post-Submission

### Preparation for Presentation
- [ ] Presentation slides prepared
- [ ] Architecture explained
- [ ] Techniques documented
- [ ] Results analyzed
- [ ] Questions anticipated

### Questionnaire
- [ ] Techniques questionnaire completed
- [ ] Experience questionnaire completed (after competition)
- [ ] Responses are clear and honest

## Troubleshooting Reference

### Common Issues
- **Connection refused**: Check server is running, correct IP/port
- **Timeout**: Reduce search depth, optimize model
- **Illegal move**: Check move generation logic
- **Model not found**: Verify model path, include in VM
- **Memory error**: Reduce transposition table size

### Emergency Fixes
- Fallback to heuristic-only evaluation
- Reduce search depth to 2-3
- Disable transposition table
- Use SimpleValueNetwork instead of ValueNetwork

## Success Criteria

Your agent is ready if:
- ✅ All local tests pass
- ✅ VM testing successful
- ✅ Moves computed within time limit
- ✅ Games complete without errors
- ✅ Submission files prepared
- ✅ Documentation complete

Good luck with the competition! 🎮

