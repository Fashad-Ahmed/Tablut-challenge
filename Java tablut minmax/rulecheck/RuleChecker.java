package it.unibo.ai.didattica.competition.tablut.rulecheck;

import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

public interface RuleChecker {

    public boolean isValid(Pawn[][] board, Action a);

    public Pawn[][] capturesWhites(Pawn[][] board, Action A);

    public Pawn[][] capturesBlacks(Pawn[][] board, Action A);

    public Pawn[][] perform(Pawn[][] board, Action a, boolean isMax,State.Turn maximizingPlayer);
} 
