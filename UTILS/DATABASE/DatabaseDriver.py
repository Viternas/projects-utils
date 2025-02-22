from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
from sqlalchemy.orm import scoped_session


class DatabaseDriver(object):
    def __init__(
            self,
            user: str = None,
            password: str = None,
            host: str = None,
            port: str = None,
            database: str = None,
            maxcon: int = 10,
            mincon: int = 1
    ):


        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.mincon = mincon
        self.maxcon = maxcon


        # Create SQLAlchemy engine with QueuePool
        self.engine = create_engine(
            self.get_connection_string(),
            pool_size=self.mincon,
            max_overflow=self.maxcon - self.mincon,
            poolclass=QueuePool,
            pool_pre_ping=True  # Ensures stale connections are removed
        )

        self.session_factory = scoped_session(sessionmaker(bind=self.engine))

    def import_config(self):

        self.user = Config().return_config_database(database_user=True)
        self.password = Config().return_config_database(database_password=True)
        self.host = Config().return_config_database(database_host=True)
        self.port = Config().return_config_database(database_port=True)
        self.database = Config().return_config_database(database_database=True)
        self.maxcon = Config().return_config_database(database_max_connection=True)
        self.mincon = Config().return_config_database(database_min_connection=True)

    def get_connection_string(self):
        return f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}'

    def execute(
            self,
            execution_string: str,
            item_tuple: tuple = None,
            commit: bool = None,
            fetch_one: bool = None,
            fetch_all: bool = None):
        try:
            Session = sessionmaker(bind=self.engine)
            session = Session()

            if item_tuple is not None:
                result = session.execute(text(execution_string), item_tuple)
            else:
                result = session.execute(text(execution_string))

            if commit:
                session.commit()

            if fetch_one:
                fetch = result.fetchone()
            elif fetch_all:
                fetch = result.fetchall()
            else:
                fetch = None

            session.close()
            return fetch

        except Exception as e:
            raise e


if __name__ == '__main__':
    print("On Main")
