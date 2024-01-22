DATABASE_HOST = "localhost"
DATABASE_NAME = "postgresCont"
DATABASE_USER = "postgres"
DATABASE_PASSWORD = "123"
DATABASE_PORT = 5433
DATABASE_CONSTRING = ("postgresql://"+DATABASE_USER
                      +":"+DATABASE_PASSWORD+"@"+DATABASE_HOST
                      +":"+str(DATABASE_PORT)+"/"+DATABASE_NAME)
