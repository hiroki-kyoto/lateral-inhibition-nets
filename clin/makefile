CC=gcc
OFL=-o
IFL=-c
EFL=-lm
TAR=lin.a
OBJ=lin.o
SRC=lin.c
lin:$(OBJ)
	$(CC) $(OFL) $(TAR) $(OBJ) $(EFL)
$(OBJ):
	$(CC) $(OFL) $(OBJ) $(IFL) $(SRC) $(EFL)
clean:
	rm *.o *.a
run:
	./$(TAR)
