package it.unibo.ai.didattica.competition.tablut.evaluator;

import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;

public class AshtonTablutEvaluator implements Evaluator{

    private static final int VICTORY = 1000;
    private static int[][] kingValues = {
        {-VICTORY,VICTORY,VICTORY,-VICTORY,-VICTORY,-VICTORY,VICTORY,VICTORY,-VICTORY},
        {VICTORY,2,3,1,-VICTORY,1,3,2,VICTORY},
        {VICTORY,2,4,0,2,0,4,2,VICTORY},
        {-VICTORY,0,2,0,1,0,2,0,-VICTORY},
        {-VICTORY,-VICTORY,2,1,0,1,2,-VICTORY,-VICTORY},
        {-VICTORY,0,2,0,1,0,2,0,-VICTORY},
        {VICTORY,2,4,0,2,0,4,2,VICTORY},
        {VICTORY,2,3,1,-VICTORY,1,3,2,VICTORY},
        {-VICTORY,VICTORY,VICTORY,-VICTORY,-VICTORY,-VICTORY,VICTORY,VICTORY,-VICTORY}
    };

    private int evaluatePawns(Pawn[][] state){
        //locate king
        int whites = 0;
        int blacks = 0;
        for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				if(state[i][j]==Pawn.WHITE || state[i][j]==Pawn.KING){
                    whites+=1;
                }
                else if (state[i][j]==Pawn.BLACK){
                    blacks+=1;
                }
			}
		}
        if(whites==0) return -VICTORY;
        if(blacks==0) return VICTORY;
        return (whites-1)*2 - blacks;
    }

    private int evaluateKing(Pawn[][] state){
        //locate king
        int x=-1,y=-1;
        for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				if(state[i][j]==Pawn.KING){
                    x = i;
                    y = j;
                    break;
                }
			}
            if(x!=-1)break;
		}

        if(x==-1) return -VICTORY;

        int value = kingValues[x][y];
        //evaluate neighbors
        return value;
    }

    @Override
    public int getWhiteScore(Pawn[][] board) {
        int pawnsValue = evaluatePawns(board);
        int kingValue = evaluateKing(board);
        int result = pawnsValue+kingValue;
        return result;
    }

    @Override
    public int getBlackScore(Pawn[][] board) {
        return getWhiteScore(board)*(-1);
    }
    
}
