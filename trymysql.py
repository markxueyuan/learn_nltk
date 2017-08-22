import mysql.connector
from mysql.connector import errorcode

query = ('select AN, HD from financial_news '
         'where NE_tag is null;')
try:
    cnx = mysql.connector.connect(user='', password='',
                              host='localhost',
                              database ='corpus')
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_CHANGE_USER_ERROR:
        print('Something is wrong with your user name or password')
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print('Database does not exist')
    else:
        print(err)
else:
    cursor = cnx.cursor()
    cursor.execute(query)

    for (AN, HD) in cursor:
        pass





# to find the data directory:
# mysql> select @@datadir;