package yes;

public class Sudoku {
    public static void solveSudoku(char[][] board) {
        if(board == null || board.length == 0)
            return;
        solve(board);
    }
    public static boolean solve(char[][] board){
        for(int i = 0; i < board.length; i++){
            for(int j = 0; j < board[0].length; j++){
                if(board[i][j] == '.'){
                    for(char c = '1'; c <= '9'; c++){//trial. Try 1 through 9
                        if(isValid(board, i, j, c)){                                
                            board[i][j] = c; //Put c for this cell
                            if(solve(board))
                                return true; //If it's the solution return true
                            else
                                board[i][j] = '.'; //Otherwise go back
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }
    private static boolean isValid(char[][] board, int row, int col, char c){
        for(int i = 0; i < 9; i++) {
            if(board[i][col] != '.' && board[i][col] == c) return false; //check row
            if(board[row][i] != '.' && board[row][i] == c) return false; //check column
            if(board[3 * (row / 3) + i / 3][ 3 * (col / 3) + i % 3] != '.' && 
board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] == c) return false; //check 3*3 block
        }
        return true;
    }
    public static void main(String[] args){
    	char[][] board={{'.','.','5',  '3','.','.',  '.','.','.'},
    			        {'8','.','.',  '.','.','.',  '.','2','.'},
    			        {'.','7','.',  '.','1','.',  '5','.','.'},
   //此数独号称世界最难数独 			        
    			        {'4','.','.',  '.','.','5',  '3','.','.'},
    			        {'.','1','.',  '.','7','.',  '.','.','6'},
    			        {'.','.','3',  '2','.','.',  '.','8','.'},
    			        
    			        {'.','6','.',  '5','.','.',  '.','.','9'},
    			        {'.','.','4',  '.','.','.',  '.','3','.'},
    			        {'.','.','.',  '.','.','9',  '7','.','.'}};
    	solveSudoku(board);
    	for(int i=0;i<9;i++){
        	for(int j=0;j<9;j++)
        		System.out.print(board[i][j]);
    		System.out.print('\n');
    	}
    }
}