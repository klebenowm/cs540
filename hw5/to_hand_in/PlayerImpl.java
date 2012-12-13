import java.util.Random;

/**
 * An implementation of the computer player.
 * 
 * @author Matthew Klebenow
 * CSL: klebenow
 * CS 540: Section 1
 */
public class PlayerImpl extends Player {
	private Color color;
	private Position bestSoFar;
	private double bestScoreSoFar;
	private boolean done;

	@Override
	public void color(Color color) {
		this.color = color;
	}

	@Override
	public void move(Color[][] board) {
	    // method initializes as not done
	    this.done = false;
	    // initialize iterative deepening search depth tracking variable
	    int depth = 0;
	    
	    // Split strategy based on player color
	    if(this.color == Color.WHITE){
	        // Player is white
	        // initialize best score to minimum (worst)
	        this.bestScoreSoFar = -Double.MAX_VALUE;
	        // continually search for best move until done
	        while(!this.done){
	            // increment depth
	            depth++;
	            // cycle legal positions
	            for(Position p : getLegalMoves(board, this.color)){
	                // initialize successor GameState state
	                GameState state = makeMove(board, p, this.color);
	                // find maximum score possible
	                double score = max_value(state, -Double.MAX_VALUE, Double.MAX_VALUE, depth);
	                // check if beats bestScoreSoFar
	                if(score > this.bestScoreSoFar){
	                    // better score
	                    // update bestScoreSoFar
	                    this.bestScoreSoFar = score;
	                    // update bestSoFar
	                    this.bestSoFar = p;
	                }
	            }
	        }	        
	    } else {
	        // Player is black
	        // initialize best score to maximum (worst)
	        this.bestScoreSoFar = Double.MAX_VALUE;
	        // continually search for best move until done
	        while(!this.done){
	            // increment depth
	            depth++;
	            // cycle legal positions
	            for(Position p : getLegalMoves(board, this.color)){
	                // initialize successor GameState state
	                GameState state = makeMove(board, p, this.color);
	                // find minimum score possible
	                double score = min_value(state, -Double.MAX_VALUE, Double.MAX_VALUE, depth);
	                // check if beats bestScoreSoFar
	                if(score < this.bestScoreSoFar){
	                    // better score
	                    // update bestScoreSoFar
	                    this.bestScoreSoFar = score;
	                    // update bestSoFar
	                    this.bestSoFar = p;
	                }
	            }
	        }
	    }
	}
	
	/**
	 * Returns maximum score possible from state by alpha-beta pruning to depth.
	 * 
	 * @param state GameState current state of game
	 * @param alpha Double alpha value for alpha-beta pruning
	 * @param beta Double beta value for alpha-beta pruning
	 * @param depth Integer depth value to search to (allows for iterative 
	 * deepening)
	 * @return The maximum score possible with the given parameters.
	 */
	private double max_value(GameState state, double alpha, double beta, int depth){
	    // visited state
	    visited(state.getBoard());
	    // check if state is terminal or depth is zero
	    if(state.getMove() == Move.GAME_OVER || depth == 0){
	        // return value of state
	        return getScore(state.getBoard()); 
	    }
	    // cycle legal positions
	    for(Position p : getLegalMoves(state.getBoard(), Color.WHITE)){
	        // with position, find successor GameState
	        GameState succ = makeMove(state.getBoard(), p, Color.WHITE);
	        // if successor is a different player
	        if(succ.getMove() != Move.WHITE){
	            // find maximum of current alpha and subsequent minimum
	            alpha = Math.max(alpha, min_value(succ, alpha, beta, depth-1));
	        } else {
	            // else find maximum of current alpha and next maximum
	            alpha = Math.max(alpha, max_value(succ, alpha, beta, depth-1));
	        }
	        // alpha pruning
	        if(alpha >= beta){
	            return beta;
	        }
	    }
	    // return
	    return alpha;
	}
	
	/**
     * Returns minimum score possible from state by alpha-beta pruning to depth.
     * 
     * @param state GameState current state of game
     * @param alpha Double alpha value for alpha-beta pruning
     * @param beta Double beta value for alpha-beta pruning
     * @param depth Integer depth value to search to (allows for iterative 
     * deepening)
     * @return The minimum score possible with the given parameters.
     */
	private double min_value(GameState state, double alpha, double beta, int depth){
	    // visited state
	    visited(state.getBoard());
	    // check if state is terminal or depth is zero
	    if(state.getMove() == Move.GAME_OVER || depth == 0){
	        // return value of state
	        return getScore(state.getBoard());
	    }
	    // cycle legal positions
	    for(Position p : getLegalMoves(state.getBoard(), Color.BLACK)){
	        // with position, find successor GameState
	        GameState succ = makeMove(state.getBoard(), p, Color.BLACK);
	        // if successor is different player
	        if(succ.getMove() != Move.BLACK){
	            // find minimum of current beta and subsequent maximum
	            beta = Math.min(beta, max_value(succ, alpha, beta, depth-1));
	        } else {
	            // else find minimum of current beta and next minimum
	            beta = Math.min(beta, min_value(succ, alpha, beta, depth-1));
	        }
	        // beta pruning
	        if(alpha >= beta){
	            return alpha;
	        }
	    }
	    // return
	    return beta;
	}

	@Override
	public Position bestSoFar() {
	    this.done = true;
		return bestSoFar;
	}
}
