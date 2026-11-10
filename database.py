import sqlite3


# Create connection
conn = sqlite3.connect(
    "agrix.db",
    check_same_thread=False
)

cursor = conn.cursor()


# ---------------- USERS TABLE ----------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    email TEXT UNIQUE,
    password TEXT
)
""")


# ---------------- PREDICTION HISTORY TABLE ----------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS prediction_history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    prediction_type TEXT,
    input_data TEXT,
    result TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()


# ---------------- REGISTER USER ----------------

def add_user(username, email, password):

    cursor.execute(
        """
        INSERT INTO users
        (username, email, password)
        VALUES (?, ?, ?)
        """,
        (username, email, password)
    )

    conn.commit()


# ---------------- LOGIN VALIDATION ----------------

def login_user(email, password):

    cursor.execute(
        """
        SELECT * FROM users
        WHERE email=?
        """,
        (email,)
    )

    data = cursor.fetchone()

    return data


# ---------------- SAVE PREDICTION ----------------

def save_prediction(
    username,
    prediction_type,
    input_data,
    result
):

    cursor.execute(
        """
        INSERT INTO prediction_history
        (username, prediction_type, input_data, result)
        VALUES (?, ?, ?, ?)
        """,
        (
            username,
            prediction_type,
            input_data,
            result
        )
    )

    conn.commit()


# ---------------- GET PREDICTION HISTORY ----------------

def get_prediction_history(username):

    cursor.execute(
        """
        SELECT prediction_type,
               input_data,
               result,
               created_at
        FROM prediction_history
        WHERE username=?
        ORDER BY id DESC
        """,
        (username,)
    )

    return cursor.fetchall()


# ---------------- CLEAR PREDICTION HISTORY ----------------

def clear_prediction_history(username):

    cursor.execute(
        """
        DELETE FROM prediction_history
        WHERE username=?
        """,
        (username,)
    )

    conn.commit()