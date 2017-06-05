CC=gcc
OFL=-o
IFL=-c
TAR=lin.a
OBJ=lin.o
SRC=lin.c
lin:$(OBJ)
	$(CC) $(OFL) $(TAR) $(OBJ)
$(OBJ):
	$(CC) $(OFL) $(OBJ) $(IFL) $(SRC)
clean:
	rm *.o *.a
run:
	./$(TAR)
