package com.example.crimeface;

import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.util.Log;

public class DatabaseHelper extends SQLiteOpenHelper {

    // Database constants
    private static final String DATABASE_NAME = "Criminallog+.db";
    private static final int DATABASE_VERSION = 1;

    // Table constants
    public static final String TABLE_NAME = "FaceData";
    public static final String COLUMN_ID = "ID";
    public static final String COLUMN_NAME = "Name";
    public static final String COLUMN_FIR = "FIRNo";
    public static final String COLUMN_CASE_DESCRIPTION = "CaseDescription";
    public static final String COLUMN_FEATURE_VECTOR = "FeatureVector";

    // SQL for creating the table
    private static final String CREATE_TABLE = "CREATE TABLE " + TABLE_NAME + " (" +
            COLUMN_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
            COLUMN_NAME + " TEXT, " +
            COLUMN_FIR + " TEXT, " +
            COLUMN_CASE_DESCRIPTION + " TEXT, " +
            COLUMN_FEATURE_VECTOR + " BLOB)";

    public DatabaseHelper(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        try {
            db.execSQL(CREATE_TABLE);
            Log.d("DatabaseHelper", "Table created successfully.");
        } catch (Exception e) {
            Log.e("DatabaseHelper", "Error creating table: ", e);
        }
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        try {
            // Drop the old table if it exists and create a new one
            db.execSQL("DROP TABLE IF EXISTS " + TABLE_NAME);
            onCreate(db);
            Log.d("DatabaseHelper", "Database upgraded successfully.");
        } catch (Exception e) {
            Log.e("DatabaseHelper", "Error upgrading database: ", e);
        }
    }

    public SQLiteDatabase getWritableDatabaseSafe() {
        SQLiteDatabase db = null;
        try {
            db = this.getWritableDatabase();
        } catch (Exception e) {
            Log.e("DatabaseHelper", "Error getting writable database: ", e);
        }
        return db;
    }

    public SQLiteDatabase getReadableDatabaseSafe() {
        SQLiteDatabase db = null;
        try {
            db = this.getReadableDatabase();
        } catch (Exception e) {
            Log.e("DatabaseHelper", "Error getting readable database: ", e);
        }
        return db;
    }
}
