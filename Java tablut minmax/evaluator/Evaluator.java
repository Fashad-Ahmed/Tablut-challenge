package it.unibo.ai.didattica.competition.tablut.evaluator;

import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;

public interface Evaluator {
    public int getWhiteScore(Pawn[][] board);
    public int getBlackScore(Pawn[][] board);
}
