import os
import psycopg


def create_table():
    with psycopg.connect("host="+os.environ['HOST']+" port="+os.environ['PORT']+" dbname="+os.environ['DBNAME']+" user="+os.environ['USER']+" password="+os.environ['PASSWORD']) as conn:
        with conn.cursor() as cur:

            cur.execute("""
                CREATE TABLE IF NOT EXISTS illegal_overtakings (
                    id serial PRIMARY KEY,
                    image_path VARCHAR(255) NOT NULL,
                    name_of_road VARCHAR(100) NOT NULL,
                    kilometric_point INT,
                    direction VARCHAR(100),
                    coordinates POINT,
                    time TIMESTAMP)
                """)

            conn.commit()


if __name__ == "__main__":
    create_table()
