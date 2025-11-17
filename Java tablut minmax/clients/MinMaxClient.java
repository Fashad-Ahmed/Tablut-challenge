package it.unibo.ai.didattica.competition.tablut.client;

import java.io.IOException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import it.unibo.ai.didattica.competition.tablut.domain.*;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.evaluator.AshtonTablutEvaluator;
import it.unibo.ai.didattica.competition.tablut.evaluator.BrandubEvaluator;
import it.unibo.ai.didattica.competition.tablut.evaluator.Evaluator;
import it.unibo.ai.didattica.competition.tablut.evaluator.ModernTablutEvaluator;
import it.unibo.ai.didattica.competition.tablut.evaluator.TablutEvaluator;
import it.unibo.ai.didattica.competition.tablut.rulecheck.RuleChecker;
import it.unibo.ai.didattica.competition.tablut.rulecheck.RuleCheckerAshton;

/**
 * 
 * @author A. Piretti, Andrea Galassi
 *
 */
public class MinMaxClient extends TablutClient {

    private class Result{
        public int score;
        public int depth_left;

        public Result(int s,int l){
            score = s;
            depth_left = l;
        }
    }

    //rule checker and board evaluator
    Evaluator evaluator;
    RuleChecker engine;

    private int game;
    private int DEPTH = 5;
    int TRESHOLD = 100;
    private List<String> citadels = new ArrayList<String>();
    

    private List<int[]> allies = new ArrayList<>();
    private List<int[]> enemies = new ArrayList<>();

    public MinMaxClient(String player, String name, int gameChosen, int timeout, String ipAddress)
            throws UnknownHostException, IOException {
        super(player, name, timeout, ipAddress);
        game = gameChosen;
    }

    public MinMaxClient(String player, String name, int timeout, String ipAddress)
            throws UnknownHostException, IOException {
        this(player, name, 4, timeout, ipAddress);
    }

    public MinMaxClient(String player, int timeout, String ipAddress) throws UnknownHostException, IOException {
        this(player, "random", 4, timeout, ipAddress);
    }

    public MinMaxClient(String player) throws UnknownHostException, IOException {
        this(player, "random", 4, 60, "localhost");
    }

    Pawn[][] deepCopyBoard(Pawn[][] board) {
        Pawn[][] copy = new Pawn[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            copy[i] = board[i].clone();
        }
        return copy;
    }
    
    
    List<Action> expandMoves(int x, int y, Pawn [][] board, Turn target) throws IOException{
        List<Action> result = new ArrayList<>();
        for(int i=0;i<9;i++){
            result.add(new Action(this.getCurrentState().getBox(x, y), this.getCurrentState().getBox(x, i), target));
            result.add(new Action(this.getCurrentState().getBox(x, y), this.getCurrentState().getBox(i, y), target));
        }
        return result;
    }    

    Result getMinMaxScore(int depth, Pawn[][] board, boolean isMax, Result alpha, Result beta) throws IOException{
        int l= 0;
        if(this.getPlayer().equals(Turn.WHITE)) l = evaluator.getWhiteScore(board);
        else l = evaluator.getBlackScore(board);

        if(depth==0 || l>=TRESHOLD || l<=(-TRESHOLD))
        {
            return new Result(l, depth);
        }
        int start = isMax ? -1000: 1000;
        Result best = new Result(start, 0);
        List<int[]> targets = new ArrayList<>();
        for(int i=0;i<9;i++){
            for(int z=0;z<9;z++){
                if(isMax){
                    if(this.getPlayer().equals(Turn.WHITE)){
                        if((board[i][z]==Pawn.WHITE || board[i][z]==Pawn.KING)){
                            targets.add(new int[]{i,z});
                        }
                    }
                    else{
                        if((board[i][z]==Pawn.BLACK)){
                            targets.add(new int[]{i,z});
                        }
                    }

                }
                else{
                    if(this.getPlayer().equals(Turn.WHITE)){
                        if((board[i][z]==Pawn.BLACK)){
                            targets.add(new int[]{i,z});
                        }
                    }
                    else{
                        if((board[i][z]==Pawn.WHITE || board[i][z]==Pawn.KING)){
                            targets.add(new int[]{i,z});
                        }                           
                    }
                }
            }
        }

        for(int[] x:targets){
            int i,j;
            i=x[0];j=x[1];

            //determine turn
            Turn t = Turn.WHITE;
            if(isMax){
                t = this.getPlayer();
            }
            else{
                if(this.getPlayer().equals(Turn.WHITE)) t = Turn.BLACK;
            }

            List<Action> actions = expandMoves(i, j, this.getCurrentState().getBoard(),t);
            for(Action a:actions){
                if(beta.score<alpha.score || (beta.score==alpha.score && beta.depth_left<=alpha.depth_left))break;
                try{
                    if(engine.isValid(board, a)){
                        Pawn[][] temp = deepCopyBoard(board);
                        temp = engine.perform(board,a,isMax,this.getPlayer());
                        Result res = getMinMaxScore(depth-1, temp, !isMax, alpha, beta);
                        if(isMax){
                            //update best
                            if(res.score>best.score){
                                best.score = res.score;
                                best.depth_left = best.depth_left;
                            }
                            else if(res.score==best.score){
                                if(res.depth_left>best.depth_left){
                                     best.depth_left = res.depth_left;
                                }
                            }
                            //update alpha
                            if(alpha.score<best.score){
                                alpha = new Result(best.score, best.depth_left);
                            }
                            else if(alpha.score==best.score){
                                alpha.depth_left = Math.max(best.depth_left, alpha.depth_left);
                            }
                        }
                        else{
                            //update score
                            if(res.score<best.score){
                                best.score = res.score;
                                best.depth_left = best.depth_left;
                            }
                            else if(res.score==best.score){
                                if(res.depth_left>best.depth_left){
                                     best.depth_left = res.depth_left;
                                }
                            }
                            //update beta
                            if(beta.score>best.score){
                                beta = new Result(best.score, best.depth_left);
                            }
                            else if(beta.score==best.score){
                                beta.depth_left = Math.max(best.depth_left, beta.depth_left);
                            }
                        }
                    }
                }
                catch(Exception e){
                    System.out.println(e);
                }
            }
        }
        return best;
    }

    Action minMaxChoice(Pawn[][] board, boolean isMax, boolean turnPlayer) throws IOException{
        List<int[]> targets = allies;
        int start = isMax ? -1000 : 1000;
        Result best = new Result(start, 0);
        List<Action> choices = new ArrayList<>();

        Turn t = Turn.WHITE;
        if(isMax){
            t = this.getPlayer();
        }
        else{
            if(this.getPlayer().equals(Turn.BLACK)) t = Turn.BLACK;
        }        

        Result alpha = new Result(-100000,0);
        Result beta =  new Result(100000,0);
        for(int[] x:targets){
            int i,j;
            i=x[0];j=x[1];
            List<Action> actions = expandMoves(i, j, this.getCurrentState().getBoard(),t);
            for(Action a:actions){
                if(beta.score<alpha.score || (beta.score==alpha.score && beta.depth_left<=alpha.depth_left))break;
                try{
                    if(engine.isValid(board,a)){
                        Pawn[][] temp = deepCopyBoard(board);
                        temp = engine.perform(board,a,isMax,this.getPlayer());
                        Result score = getMinMaxScore(DEPTH-1, temp, !isMax, alpha, beta);
                        if(isMax){
                            //upadte best
                            if(best.score<score.score){
                                best.score=score.score;
                                best.depth_left = score.depth_left;
                                choices.clear();
                                choices.add(a);
                            }
                            else if(best.score==score.score){
                                if(best.depth_left<score.depth_left){
                                    best.depth_left = score.depth_left;
                                    choices.clear();
                                    choices.add(a);
                                }
                                else if(best.depth_left==score.depth_left){
                                    choices.add(a);
                                }
                            }
                            //update alpha
                            if(alpha.score<best.score){
                                alpha = new Result(best.score, best.depth_left);
                            }
                            else if(alpha.score==best.score){
                                alpha.depth_left = Math.max(best.depth_left, alpha.depth_left);
                            }
                        }
                        else{
                            //update best
                            if(best.score>score.score){
                                best.score=score.score;
                                best.depth_left = score.depth_left;
                                choices.clear();
                                choices.add(a);
                            }
                            else if(best.score==score.score){
                                if(best.depth_left<score.depth_left){
                                    best.depth_left = score.depth_left;
                                    choices.clear();
                                    choices.add(a);
                                }
                                else if(best.depth_left==score.depth_left){
                                    choices.add(a);
                                }
                            }
                            //update beta
                            if(beta.score>best.score){
                                beta = new Result(best.score, best.depth_left);
                            }
                            else if(beta.score==best.score){
                                beta.depth_left = Math.max(best.depth_left, beta.depth_left);
                            }
                        }                       
                    }
                }
                catch(Exception e){
                    System.out.println(e.toString());
                }
            }
        }
        Random s = new Random();
        System.out.println("Score obtained:"+best.score+", depth_left="+best.depth_left);
        System.out.println("Choises size:"+choices.size());
        return choices.get(s.nextInt(choices.size()));
    }

    public static void main(String[] args) throws UnknownHostException, IOException, ClassNotFoundException {
        int gametype = 4;
        String role = "";
        String name = "minmax-bro";
        String ipAddress = "localhost";
        int timeout = 60;
        // TODO: change the behavior?
        if (args.length < 1) {
            System.out.println("You must specify which player you are (WHITE or BLACK)");
            System.exit(-1);
        } else {
            System.out.println(args[0]);
            role = (args[0]);
        }
        if (args.length == 2) {
            System.out.println(args[1]);
            timeout = Integer.parseInt(args[1]);
        }
        if (args.length == 3) {
            ipAddress = args[2];
        }
        System.out.println("Selected client: " + args[0]);

        MinMaxClient client = new MinMaxClient(role, name, gametype, timeout, ipAddress);
        client.run();
        if(args[0].toLowerCase()=="black"){
            client.setPlayer(Turn.BLACK);
        }
        else{
            client.setPlayer(Turn.WHITE);
        }
    }

    @Override
    public void run() {
        try {
            this.declareName();
        } catch (Exception e) {
            e.printStackTrace();
        }

        State state;

        switch (this.game) {
            case 1:
                state = new StateTablut();
                evaluator = new TablutEvaluator();
                break;
            case 2:
                state = new StateTablut();
                evaluator = new ModernTablutEvaluator();
                break;
            case 3:
                state = new StateBrandub();
                evaluator = new BrandubEvaluator();
                break;
            case 4:
                this.citadels.add("a4");
                this.citadels.add("a5");
                this.citadels.add("a6");
                this.citadels.add("b5");
                this.citadels.add("d1");
                this.citadels.add("e1");
                this.citadels.add("f1");
                this.citadels.add("e2");
                this.citadels.add("i4");
                this.citadels.add("i5");
                this.citadels.add("i6");
                this.citadels.add("h5");
                this.citadels.add("d9");
                this.citadels.add("e9");
                this.citadels.add("f9");
                this.citadels.add("e8");
                state = new StateTablut();
                evaluator = new AshtonTablutEvaluator();
                engine = new RuleCheckerAshton(citadels);
                System.out.println("Ashton Tablut game");
                break;
            default:
                System.out.println("Error in game selection");
                System.exit(4);
        }

        System.out.println("You are player " + this.getPlayer().toString() + "!");
        while (true) {
            try {
                this.read();
            } catch (ClassNotFoundException | IOException e1) {
                // TODO Auto-generated catch block
                e1.printStackTrace();
                System.exit(1);
            }
            System.out.println("Current state:");
            state = this.getCurrentState();
            System.out.println(state.toString());

            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
            }

            allies.clear();
            enemies.clear();
            if (this.getPlayer().equals(Turn.WHITE)) {
                //load pawns
                for (int i = 0; i < 9; i++) {
                    for (int j = 0; j < 9; j++) {
                        if (state.getPawn(i, j).equalsPawn(State.Pawn.WHITE.toString())
                                || state.getPawn(i, j).equalsPawn(State.Pawn.KING.toString())) {
                            int[] buf = new int[2];
                            buf[0] = i;
                            buf[1] = j;
                            allies.add(buf);
                        }
                        else if(state.getPawn(i, j).equalsPawn(State.Pawn.BLACK.toString())){
                            int[] buf = new int[2];
                            buf[0] = i;
                            buf[1] = j;
                            enemies.add(buf);
                        }
                    }
                }

                // Mio turno
                if (this.getCurrentState().getTurn().equals(StateTablut.Turn.WHITE)) {
                    // TODO: implement variations
                    Action a = null;
                    try {
                        a = minMaxChoice(this.getCurrentState().getBoard(), true, true);
                        System.out.println("Mossa scelta: " + a.toString());
                    } catch (IOException e) {
                        // TODO Auto-generated catch block
                        System.out.println(e.toString());
                    }
                    
                    try {
                        this.write(a);
                    } catch (ClassNotFoundException | IOException e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                    }

                }
                // Turno dell'avversario
                else if (state.getTurn().equals(StateTablut.Turn.BLACK)) {
                    System.out.println("Waiting for your opponent move... ");
                }
                // ho vinto
                else if (state.getTurn().equals(StateTablut.Turn.WHITEWIN)) {
                    System.out.println("YOU WIN!");
                    System.exit(0);
                }
                // ho perso
                else if (state.getTurn().equals(StateTablut.Turn.BLACKWIN)) {
                    System.out.println("YOU LOSE!");
                    System.exit(0);
                }
                // pareggio
                else if (state.getTurn().equals(StateTablut.Turn.DRAW)) {
                    System.out.println("DRAW!");
                    System.exit(0);
                }

            } else {

                // Mio turno
                if (this.getCurrentState().getTurn().equals(StateTablut.Turn.BLACK)) {

                for (int i = 0; i < 9; i++) {
                    for (int j = 0; j < 9; j++) {
                       if (state.getPawn(i, j).equalsPawn(State.Pawn.WHITE.toString())
                                || state.getPawn(i, j).equalsPawn(State.Pawn.KING.toString())) {
                            int[] buf = new int[2];
                            buf[0] = i;
                            buf[1] = j;
                            enemies.add(buf);
                        }
                        else if(state.getPawn(i, j).equalsPawn(State.Pawn.BLACK.toString())){
                            int[] buf = new int[2];
                            buf[0] = i;
                            buf[1] = j;
                            allies.add(buf);
                        }
                    }
                }
                    // TODO: implement variations
                    Action a = null;
                    try {
                        a = minMaxChoice(this.getCurrentState().getBoard(), true, true);
                    } catch (IOException e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                    }
                    System.out.println("Mossa scelta: " + a.toString());
                    try {
                        this.write(a);
                    } catch (ClassNotFoundException | IOException e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                    }
                }

                else if (state.getTurn().equals(StateTablut.Turn.WHITE)) {
                    System.out.println("Waiting for your opponent move... ");
                } else if (state.getTurn().equals(StateTablut.Turn.WHITEWIN)) {
                    System.out.println("YOU LOSE!");
                    System.exit(0);
                } else if (state.getTurn().equals(StateTablut.Turn.BLACKWIN)) {
                    System.out.println("YOU WIN!");
                    System.exit(0);
                } else if (state.getTurn().equals(StateTablut.Turn.DRAW)) {
                    System.out.println("DRAW!");
                    System.exit(0);
                }

            }
        }

    }
}
