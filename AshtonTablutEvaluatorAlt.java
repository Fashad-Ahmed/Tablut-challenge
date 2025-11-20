package it.unibo.ai.didattica.competition.tablut.evaluator;

import java.util.ArrayList;
import java.util.List;

import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;


public class AshtonTablutEvaluatorAlt implements Evaluator{

    private static final int VICTORY = 1000;
    private static int[][] kingValues = {
        {-VICTORY,VICTORY,VICTORY,-VICTORY,-VICTORY,-VICTORY,VICTORY,VICTORY,-VICTORY},
        {VICTORY,0,3,0,-VICTORY,0,3,0,VICTORY},
        {VICTORY,0,4,0,2,0,4,0,VICTORY},
        {-VICTORY,0,2,0,1,0,2,0,-VICTORY},
        {-VICTORY,-VICTORY,2,1,0,1,2,-VICTORY,-VICTORY},
        {-VICTORY,0,2,0,1,0,2,0,-VICTORY},
        {VICTORY,0,4,0,2,0,4,0,VICTORY},
        {VICTORY,0,3,0,-VICTORY,0,3,0,VICTORY},
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
        int[][] neigPos = {{x-1,y},{x,y-1},{x+1,y},{x,y+1}};
        List<int[]> neig = new ArrayList<>();

        for(int i = 0; i < 4; i++)
            if(neigPos[i][0] > -1 && neigPos[i][1] > -1 && neigPos[i][0] < kingValues.length && neigPos[i][1] < kingValues.length)
                neig.add(neigPos[i]);
        
        for(int i = 0;i < neig.size(); i++)
            if(state[neig.get(i)[0]][neig.get(i)[1]] == Pawn.BLACK || kingValues[neig.get(i)[0]][neig.get(i)[1]] == -VICTORY ) value--;
        
            
        //evaluate free paths
        for(int i = 0;i < neig.size(); i++){
            int countPosX = 0, countPosY = 0, deltaX = neig.get(i)[0] - x, deltaY = neig.get(i)[1] - y;

            while (true) {
                countPosX += deltaX; 
                countPosY += deltaY;
                
                int kingVal = kingValues[x + countPosX][y + countPosY];
                Pawn currentState = state[x + countPosX][y + countPosY]; 
                if(kingVal == VICTORY){
                    value++;
                    break;
                }else if(kingVal == -VICTORY 
                || currentState == Pawn.BLACK
                || currentState == Pawn.WHITE) break;
            }
        }
        
        
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

