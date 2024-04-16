from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse
from fastapi import Query
import os
from typing import Optional
from sqlalchemy import func
from jose import jwt
from pytz import timezone
from fastapi.middleware.cors import CORSMiddleware
import pytz
import pandas as pd


#追加分2023/03/12
from fastapi import status
import re


# 環境変数の読み込み
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')

# トークンを生成するためのシークレットキー
SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = "HS256"

app = FastAPI()

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ここに許可するオリジンを指定します。*はすべてのオリジンを許可します。
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # 許可するメソッドを指定します。
    allow_headers=["*"],  # すべてのヘッダーを許可します。必要に応じて指定します。
)

# データベース接続設定
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ProductDB(Base):
    __tablename__ = 'products'
    product_id = Column(Integer, primary_key=True, nullable=False)
    product_name = Column(String(255), nullable=False)
    price = Column(Integer, nullable=False)
    product_qrcode = Column(Integer, unique=True, nullable=False)
    quantity = Column(Integer, nullable=False)
    last_update = Column(DateTime, default=datetime.utcnow)

class UserDB(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True, nullable=False)
    user_name = Column(String(255), nullable=False)
    birthplace = Column(String(255), nullable=False)
    password = Column(String(255), nullable=True)
    token = Column(String(500), nullable=True)
    last_update = Column(DateTime, default=datetime.utcnow)

# ユーザー情報を格納するためのモデル
class User(BaseModel):
    user_name: str
    password: str

class Deal_DetailsDB(Base):
    __tablename__ = 'deal_details'  # テーブル名を指定
    deal_id = Column(Integer, primary_key=True, nullable=False)
    quantity = Column(Integer, nullable=False)
    barcode = Column(Integer, nullable=False)
    product_name = Column(String(255), nullable=False)  # 長さを255に合わせる
    price = Column(Integer, nullable=False)
    peer = Column(Integer, nullable=False)
    tax_percent = Column(Numeric(precision=5, scale=2), nullable=False)
    buy_time = Column(DateTime, nullable=False)  # nullable=TrueからFalseに変更

# リクエストボディのモデル定義
class Product(BaseModel):
    barcode: int
    product_name: str
    quantity: int
    price: int
    peer: int
    tax_percent: float
    buy_time: datetime

class ProductList(BaseModel):
    products: list[Product]


class TaxDB(Base):
    __tablename__ = 'tax'  # テーブル名を指定
    tax_id = Column(Integer, primary_key=True, nullable=False)
    tax_code = Column(Integer, unique=True, nullable=False)
    tax_name = Column(String(255), nullable=False)  # 長さを255に合わせる
    tax_percent = Column(Numeric(precision=5, scale=2), nullable=False) # MySQLのdecimal型に対応するためにNumeric型を使用

# SQLAlchemyのテーブルモデル
class TradesDB(Base):
    __tablename__ = 'trades'
    trade_id = Column(Integer, primary_key=True, autoincrement=True)
    buy_time = Column(DateTime, default=func.now(), nullable=False)
    staff_id = Column(Integer, nullable=False)
    machine_id = Column(Integer, nullable=False)
    store_id = Column(Integer, nullable=False)
    total_charge = Column(Integer, nullable=False)
    total_charge_wo_tax = Column(Integer, nullable=False)
    total_peer = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=True)

# Pydanticモデルの定義
class Trades(BaseModel):
    token: str
    store_id: int
    staff_id: int
    machine_id: int
    total_charge: int
    total_charge_wo_tax: int
    total_peer: int
    buy_time: Optional[datetime]  # buy_time フィールドをオプションに変更


class EventsDB(Base):
    __tablename__ = 'events'
    event_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    event_title = Column(String(50), nullable=False)
    host_name = Column(String(50), nullable=False)
    event_time = Column(DateTime, nullable=True)


class RegistrationsDB(Base):
    __tablename__ = 'registrations'
    registration_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    user_name = Column(String(45), nullable=False)
    product_name = Column(String(45), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Integer, nullable=False)
    peer = Column(Integer, nullable=False)
    now_counts = Column(Integer, nullable=False)
    initial_counts = Column(Integer, nullable=False)
    barcode = Column(Integer, nullable=False)
    message = Column(String(200), nullable=True)
    range_name = Column(String(45), nullable=True)
    registration_date = Column(DateTime, nullable=False)
    last_update = Column(DateTime, nullable=False)

class Registrations(BaseModel):
    token: str
    product_name: str
    price: int
    peer: int
    initial_counts: int
    message: str
    range_name: str
    registration_date: Optional[datetime]



class FriendsDB(Base):
    __tablename__ = 'friends'
    friend_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    my_name = Column(String(45), nullable=False)
    friend_name = Column(String(45), nullable=False)
    approval_date = Column(DateTime, nullable=True)


class ParticipantsDB(Base):
    __tablename__ = 'participants'
    participant_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    event_id = Column(Integer, nullable=False)
    participant_name = Column(String(45), nullable=False)
    comment = Column(String(200), nullable=False)
    last_update = Column(DateTime, nullable=True)


class RangesDB(Base):
    __tablename__ = 'ranges'
    range_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    range_name = Column(String(45), nullable=False)


class VegetablesDB(Base):
    __tablename__ = 'vegetables'
    vegetable_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    vegetable_name = Column(String(45), nullable=False)


class Legacy_UsersDB(Base):
    __tablename__ = 'legacy_users'
    legacy_users_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False, unique=True)
    legacy_user_name = Column(String(45), nullable=False)
    family_id = Column(Integer, nullable=False)
    relationship = Column(String(45), nullable=False)
    complex = Column(String(45), nullable=False)
    wing = Column(String(45), nullable=False)
    floor = Column(String(45), nullable=False)
    registration_date = Column(DateTime, nullable=False)



# データベースのセットアップ
Base.metadata.create_all(bind=engine)


# UTCで取得した日時をJSTに変換する関数
def to_jst(datetime_obj):
    utc_zone = pytz.utc
    jst_zone = pytz.timezone('Asia/Tokyo')
    return datetime_obj.replace(tzinfo=utc_zone).astimezone(jst_zone)


def calculate_luhn_checksum(barcode_str):
    # 各桁の数字に対して逆順に処理を行うための準備
    digits = [int(digit) for digit in reversed(barcode_str)]

    # 合計を計算するための変数
    total_sum = 0

    # 各桁について処理
    for i, digit in enumerate(digits):
        # 奇数番目（実際には偶数インデックス）の場合はそのまま、偶数番目（奇数インデックス）の場合は2倍
        if i % 2 == 0:
            # 奇数番目の桁はそのまま加算
            total_sum += digit
        else:
            # 偶数番目の桁は2倍して、10以上なら各桁の和を加算
            doubled_digit = digit * 2
            total_sum += doubled_digit if doubled_digit < 10 else doubled_digit - 9

    # チェックディジットの計算（10から合計の1の位を引く。ただし、結果が10の場合は0とする）
    check_digit = (10 - (total_sum % 10)) % 10

    return check_digit

def extract_numeric_timestamp(registration_date):
    # datetimeオブジェクトをISOフォーマットの文字列に変換
    if isinstance(registration_date, datetime):
        registration_date_str = registration_date.isoformat()
    else:
        registration_date_str = str(registration_date)

    # 正規表現を使用して数字のみを抽出
    numeric_parts = re.findall(r'\d', registration_date_str)

    # 抽出した数字を結合して一つの文字列にし、整数に変換
    numeric_timestamp = int(''.join(numeric_parts))//1000000000000
    print(numeric_timestamp)
    return numeric_timestamp

def remove_milliseconds(dt):
    # マイクロ秒部分を0に設定してミリ秒を削除
    return dt.replace(microsecond=0)


@app.post('/login')
async def login(user: User):
    db = SessionLocal()
    # ユーザーの認証
    user_info = db.query(UserDB).filter_by(user_name=user.user_name, password=user.password).first()
    if not user_info:
        raise HTTPException(status_code=401, detail="Bad username or password")

    now = datetime.now()

    # トークンが有効期限内であるかどうかをチェックし、トークンを発行または更新する
    if user_info.last_update is None or (now - user_info.last_update) > timedelta(days=7):
        # トークンのペイロード
        payload = {
            "sub": user_info.user_name,
            "exp": now + timedelta(days=7)  # トークンの有効期限を7日に設定
        }

        # トークンを生成
        access_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

        # ユーザー情報の更新
        user_info.token = access_token
        user_info.last_update = now
        db.commit()
    else:
        access_token = user_info.token

    # トークンをクライアントに返す
    return {"access_token": access_token, "user_name": user_info.user_name}


@app.get('/shopping')
async def read_login(token: str = Query(..., description="Token information")):
    db = SessionLocal()
    # ユーザーの認証
    user_info = db.query(UserDB).filter_by(token=token).first()
    if not user_info:
        raise HTTPException(status_code=401, detail="Bad token")

    user_name = user_info.user_name

    # ユーザー名をクライアントに返す
    return {"user_name": user_name}


# @app.get("/qrcode")
# async def read_products_info(qrcode: int = Query(..., description="Product QR code")):
#     db = SessionLocal()
#     product = db.query(ProductDB).filter_by(product_qrcode=qrcode).first()
#     if product:
#         # Productの情報を取得
#         product_info = {
#             "product_id": product.product_id,
#             "product_name": product.product_name,
#             "price": product.price,
#             "product_qrcode": product.product_qrcode,
#             "quantity": product.quantity,
#         }

#         # Taxの情報を取得
#         tax = db.query(TaxDB).first()
#         if tax:
#             product_info["tax_percent"] = tax.tax_percent
#         else:
#             product_info["tax_percent"] = 0.1  # デフォルト値などを設定する必要がある場合

#         db.close()
#         return product_info
#     else:
#         db.close()
#         return JSONResponse(content={"product_name": "商品がマスタ未登録です"}, status_code=404)


@app.post('/trade')
async def add_trade(trade: Trades):
    db = SessionLocal()
    # ユーザーの情報を取得
    user_info = db.query(UserDB).filter_by(token=trade.token).first()
    if not user_info:
        raise HTTPException(status_code=404, detail="User not found")

    # UTCで取得した日時をJSTに変換
    buy_time_utc = remove_milliseconds(product.buy_time)
    jst = pytz.timezone('Asia/Tokyo')
    buy_time_jst = buy_time_utc.astimezone(jst)

    new_trade = TradesDB(
        user_id=user_info.user_id,
        store_id=trade.store_id,
        staff_id=trade.staff_id,
        machine_id=trade.machine_id,
        total_charge=trade.total_charge,
        total_charge_wo_tax=trade.total_charge_wo_tax,
        total_peer=trade.total_peer,
        buy_time=buy_time_jst
    )

    # トレードをデータベースに追加してコミット
    db.add(new_trade)
    db.commit()

    # 成功した場合は挿入されたトレードのIDを含むレスポンスを返す
    return {"trade_id": new_trade.trade_id}


# FastAPIのエンドポイント
@app.post('/deal_detail')
def add_deal_detail(products: ProductList):
    db = SessionLocal()
    jst = pytz.timezone('Asia/Tokyo')  # 日本時間のタイムゾーンを設定

    for product in products.products:
        # UTCで取得した日時をJSTに変換
        buy_time_utc = remove_milliseconds(product.buy_time)
        buy_time_jst = buy_time_utc.astimezone(jst)

        new_detail = Deal_DetailsDB(
            barcode=product.barcode,
            product_name=product.product_name,
            price=product.price,
            peer=product.peer,
            quantity=product.quantity,
            tax_percent=product.tax_percent,
            buy_time=buy_time_jst  # JSTに変換した日時を使用
        )
        db.add(new_detail)
    db.commit()
    return {'message': 'Deal details added successfully'}


#登録画面に遷移した際に、登録できる野菜をNETX.jsに渡す
@app.get("/vegetables")
async def read_vegetables_info(skip: int = 0, limit: int = 99):
    db = SessionLocal()
    vegetables = db.query(VegetablesDB).offset(skip).limit(limit).all()
    return {"vegetables": [vegetable.vegetable_name for vegetable in vegetables]}

#登録画面に遷移した際に、登録できるrangeをNETX.jsに渡す
@app.get("/ranges")
async def read_ranges_info(skip: int = 0, limit: int = 99):
    db = SessionLocal()
    ranges = db.query(RangesDB).offset(skip).limit(limit).all()
    return {"range": [range.range_name for range in ranges]}


@app.post("/registrations", status_code=status.HTTP_201_CREATED)
def create_registration(registration: Registrations):
    db = SessionLocal()
    # tokenからuser_nameを取得
    user = db.query(UserDB).filter(UserDB.token == registration.token).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # vegetable_nameからvegetable_idを取得してbarcodeを生成
    vegetable = db.query(VegetablesDB).filter(VegetablesDB.vegetable_name == registration.product_name).first()
    if vegetable is None:
        raise HTTPException(status_code=404, detail="Vegetable not found")

    user_str = str(user.user_id).zfill(3)
    registration_date_str = str(extract_numeric_timestamp(registration.registration_date)).zfill(12)

    # バーコード生成ロジック (チェックディジットを計算する方法を追加する必要があります)
    barcode_str = registration_date_str + user_str
    # ここにチェックディジットの計算と追加のロジックを実装

    # チェックディジットの計算 (Luhnアルゴリズム)
    check_digit = calculate_luhn_checksum(barcode_str)
    barcode = int(barcode_str + str(check_digit))

    db_registration = RegistrationsDB(
        user_name=user.user_name,
        product_name=registration.product_name,
        quantity=1,
        price=registration.price,
        peer=registration.peer,
        now_counts=registration.initial_counts,
        initial_counts=registration.initial_counts,
        message=registration.message,
        range_name=registration.range_name,
        registration_date=registration.registration_date,
        last_update=registration.registration_date,
        barcode=barcode
    )
    db.add(db_registration)
    db.commit()
    db.refresh(db_registration)
    return db_registration


@app.get("/print")
async def read_registration_info(registration_id: int = Query(..., description="Registration Id")):
    db = SessionLocal()
    printing = db.query(RegistrationsDB).filter_by(registration_id=registration_id).first()
    if printing:
        # printingの情報を取得
        printing_info = {
            "product_name": printing.product_name,
            "price": printing.price,
            "peer": printing.peer,
            "user_name": printing.user_name,
            "barcode": printing.barcode,
        }

        db.close()
        return printing_info
    else:
        db.close()
        return JSONResponse(content={"product_id": "idが未登録です"}, status_code=404)

@app.get("/barcode")
async def read_registrations_info(barcode: int = Query(..., description="Product barcode")):
    db = SessionLocal()
    product = db.query(RegistrationsDB).filter_by(barcode=barcode).first()
    if product:
        # Productの情報を取得
        product_info = {
            "registration_id": product.registration_id,
            "product_name": product.product_name,
            "price": product.price,
            "peer": product.peer,
            "barcode": product.barcode,
            "quantity": product.quantity,
        }

        # Taxの情報を取得
        tax = db.query(TaxDB).first()
        if tax:
            product_info["tax_percent"] = tax.tax_percent
        else:
            product_info["tax_percent"] = 0.1  # デフォルト値などを設定する必要がある場合

        db.close()
        return product_info
    else:
        db.close()
        return JSONResponse(content={"product_name": "商品がマスタ未登録です"}, status_code=404)


@app.get("/trade")
async def read_trade_info(
    buy_time: str = Query(..., description="Time of purchase", example="2024-04-14T10:09:00.490Z")
):

    db = SessionLocal()

    try:
        # buy_timeをdatetimeに変換

        utc_time = datetime.fromisoformat(buy_time.rstrip("Z")).replace(tzinfo=pytz.utc)

        # 日本時間のタイムゾーンを設定
        jst_timezone = pytz.timezone("Asia/Tokyo")

        # UTCから日本時間に変換
        jst_time = utc_time.astimezone(jst_timezone)

        buy_time_date = jst_time.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        db.close()
        raise HTTPException(status_code=400, detail="Invalid buy_time format. Please use YYYY-MM-DD HH:MM:SS format.")

    trade = db.query(TradesDB).filter_by(
            buy_time=buy_time_date,
        ).first()

    if trade:
        # tradeの情報を取得
        trade_result = {
            "store_id": trade.store_id,
            "staff_id": trade.staff_id,
            "machine_id": trade.machine_id,
            "total_charge": trade.total_charge,
            "total_charge_wo_tax": trade.total_charge_wo_tax,
            "total_peer": trade.total_peer,
        }

        db.close()
        return trade_result
    else:
        db.close()
        return JSONResponse(content={"message": "購買履歴がありません"}, status_code=404)


@app.get("/deal_detail")
async def read_deal_detail_info(
    buy_time: str = Query(..., description="Time of purchase", example="2024-04-14T10:09:00.490Z")
):
    db = SessionLocal()

    try:
        # buy_timeをdatetimeに変換
        utc_time = datetime.fromisoformat(buy_time.rstrip("Z")).replace(tzinfo=pytz.utc)

        # 日本時間のタイムゾーンを設定
        jst_timezone = pytz.timezone("Asia/Tokyo")

        # UTCから日本時間に変換
        jst_time = utc_time.astimezone(jst_timezone)

        # buy_timeをdatetimeに変換
        buy_time_date = jst_time.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        db.close()
        raise HTTPException(status_code=400, detail="Invalid buy_time format. Please use YYYY-MM-DD HH:MM:SS format.")

    # Deal_DetailsDB と RegistrationsDB を barcode で結合
    deal_details = db.query(
        Deal_DetailsDB.product_name,
        Deal_DetailsDB.quantity,
        Deal_DetailsDB.price,
        Deal_DetailsDB.peer,
        Deal_DetailsDB.tax_percent,
        RegistrationsDB.user_name
    ).join(
        RegistrationsDB, Deal_DetailsDB.barcode == RegistrationsDB.barcode
    ).filter(
        Deal_DetailsDB.buy_time == buy_time_date
    ).all()

    if deal_details:
        # deal_detailsの情報を取得
        deal_result = [
            {
            "product_name": deal_detail.product_name,
            "quantity": deal_detail.quantity,
            "price": deal_detail.price,
            "peer": deal_detail.peer,
            "tax_percent": float(deal_detail.tax_percent),
            "user_name": deal_detail.user_name if deal_detail.user_name is not None else "-"
            }
            for deal_detail in deal_details
        ]

        db.close()
        return {"deal_details": deal_result}
    else:
        db.close()
        return JSONResponse(content={"deal_details": "購買履歴がありません"}, status_code=404)


@app.get("/message")
async def read_message_info(
    buy_time: str = Query(..., description="Time of purchase", example="2024-04-14T10:09:00.490Z")
):
    db = SessionLocal()

    try:
        # buy_timeをdatetimeに変換
        utc_time = datetime.fromisoformat(buy_time.rstrip("Z")).replace(tzinfo=pytz.utc)

        # 日本時間のタイムゾーンを設定
        jst_timezone = pytz.timezone("Asia/Tokyo")

        # UTCから日本時間に変換
        jst_time = utc_time.astimezone(jst_timezone)

        # buy_timeをdatetimeに変換
        buy_time_date = jst_time.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        db.close()
        raise HTTPException(status_code=400, detail="Invalid buy_time format. Please use YYYY-MM-DD HH:MM:SS format.")

    # Deal_DetailsDB と RegistrationsDB を barcode で結合し、RegistrationsDBとLegacy_UsersDBをuser_name==legacy_user_nameで統合
    buyer_infos = db.query(
        Deal_DetailsDB.deal_id,
        RegistrationsDB.user_name,
        RegistrationsDB.range_name,
        RegistrationsDB.message,
        Legacy_UsersDB.legacy_user_name,
        Legacy_UsersDB.complex,
        Legacy_UsersDB.wing,
        Legacy_UsersDB.floor,
        Legacy_UsersDB.family_id,
    ).join(
        RegistrationsDB, Deal_DetailsDB.barcode == RegistrationsDB.barcode
        ).join(
        Legacy_UsersDB, RegistrationsDB.user_name == Legacy_UsersDB.legacy_user_name
    ).filter(
        Deal_DetailsDB.buy_time == buy_time_date
    ).all()

    if buyer_infos:
        # buyer_infosの情報を取得
        buyer_info_result = [
            {
            "deal_id": buyer_info.deal_id if buyer_info.deal_id is not None else "-",
            "buyer_name": buyer_info.user_name if buyer_info.user_name is not None else "-",
            "range_name": buyer_info.range_name if buyer_info.range_name is not None else "-",
            "message": buyer_info.message if buyer_info.message is not None else "-",
            "buyer_complex": buyer_info.complex if buyer_info.complex is not None else "-",
            "buyer_wing": buyer_info.wing if buyer_info.wing is not None else "-",
            "buyer_floor": buyer_info.floor if buyer_info.floor is not None else "-",
            "buyer_family_id": buyer_info.family_id if buyer_info.family_id is not None else "-"
            }
            for buyer_info in buyer_infos
        ]

        # Deal_DetailsDB と TradesDB を buy_time で結合し、TradesDBとUserDBをuser_idで統合し、UsersDBとLegacy_UsersDBをuser_name==legacy_user_nameで統合
    seller_infos = db.query(
        Deal_DetailsDB.deal_id,
        Legacy_UsersDB.legacy_user_name,
        Legacy_UsersDB.complex,
        Legacy_UsersDB.wing,
        Legacy_UsersDB.floor,
        Legacy_UsersDB.family_id,
    ).join(
        TradesDB, Deal_DetailsDB.buy_time == TradesDB.buy_time
        ).join(
        UserDB, TradesDB.user_id == UserDB.user_id
        ).outerjoin(
        Legacy_UsersDB, UserDB.user_name == Legacy_UsersDB.legacy_user_name
    ).filter(
        Deal_DetailsDB.buy_time == buy_time_date
    ).all()

    if seller_infos:
        # buyer_infosの情報を取得
        seller_info_result = [
            {
            "deal_id": seller_info.deal_id if seller_info.deal_id is not None else "-",
            "seller_name": seller_info.legacy_user_name if seller_info.legacy_user_name is not None else "-",
            "seller_complex": seller_info.complex if seller_info.complex is not None else "-",
            "seller_wing": seller_info.wing if seller_info.wing is not None else "-",
            "seller_floor": seller_info.floor if seller_info.floor is not None else "-",
            "seller_family_id": seller_info.family_id if seller_info.family_id is not None else "-"
            }
            for seller_info in seller_infos
        ]

        db.close()


    # データを pandas DataFrame に変換
    if buyer_infos:
        buyer_df = pd.DataFrame(
                [info for info in buyer_infos],
                columns=[
                    "deal_id",
                    "buyer_name",
                    "range_name",
                    "message",
                    "buyer_legacy_user_name",
                    "buyer_complex",
                    "buyer_wing",
                    "buyer_floor",
                    "buyer_family_id"
                ]
            )
    else:
        buyer_df = pd.DataFrame()

    if seller_infos:
        seller_df = pd.DataFrame(
                [info for info in seller_infos],
                columns=[
                    "deal_id",
                    "seller_name",
                    "seller_complex",
                    "seller_wing",
                    "seller_floor",
                    "seller_family_id"
                ]
            )
    else:
        seller_df = pd.DataFrame()

    # DataFrame を 'deal_id' で結合
    combined_df = pd.merge(buyer_df, seller_df, on="deal_id", how="outer")

    # 'range_name' が "メッセージを送らない" と等しくない行だけを保持
    combined_df = combined_df[combined_df['range_name'] != "メッセージを送らない"]

    # 条件1: "range_name"が"同じ団地の方へ"で、buyer_complexとseller_complexが異なる行を削除
    mask1 = (combined_df['range_name'] == "同じ団地の方へ") & (combined_df['buyer_complex'] != combined_df['seller_complex'])
    combined_df = combined_df[~mask1]

    # 条件2: "range_name"が"同じ棟の方へ"で、同じcomplex内で異なるwingの行を削除
    mask2 = (combined_df['range_name'] == "同じ棟の方へ") & ((combined_df['buyer_complex'] != combined_df['seller_complex']) | (combined_df['buyer_wing'] != combined_df['seller_wing']))
    combined_df = combined_df[~mask2]

    # データ型を確認し、必要に応じて整数型に変換
    try:
        combined_df['buyer_floor'] = pd.to_numeric(combined_df['buyer_floor'], errors='coerce')
        combined_df['seller_floor'] = pd.to_numeric(combined_df['seller_floor'], errors='coerce')
    except Exception as e:
        return {"error": str(e)}

    # データ型変換後に None が含まれているか確認
    if combined_df['buyer_floor'].isnull().any() or combined_df['seller_floor'].isnull().any():
        return {"error": "Floor data contains non-integer values or NaN."}

    # 条件3: "range_name"が"同じ棟の同じ階の方へ"で、同じcomplex, 同じwingで異なるfloorの行を削除
    # floorの10の位を取得するために、floorを整数型として扱い、10で割った商を比較
    combined_df['buyer_floor_tens'] = combined_df['buyer_floor'] // 10
    combined_df['seller_floor_tens'] = combined_df['seller_floor'] // 10

    mask3 = (
        combined_df['range_name'] == "同じ棟の同じ階の方へ") & (
        (combined_df['buyer_complex'] != combined_df['seller_complex']) |
        (combined_df['buyer_wing'] != combined_df['seller_wing']) |
        (combined_df['buyer_floor_tens'] != combined_df['seller_floor_tens'])
    )
    combined_df = combined_df[~mask3]

    # 不要になった 'buyer_floor_tens' と 'seller_floor_tens' 列を削除
    combined_df.drop(['buyer_floor_tens', 'seller_floor_tens'], axis=1, inplace=True)

    # 'buyer_name', 'range_name', 'seller_name', 'message' の組み合わせが重複している行を識別
    duplicates = combined_df.duplicated(subset=['buyer_name', 'range_name', 'seller_name', 'message'], keep=False)

    # 重複している行を削除
    combined_df = combined_df[~duplicates]

    # 'range_name' カラムの値から "の方へ" という文字列を削除
    combined_df['range_name'] = combined_df['range_name'].str.replace("の方へ", "", regex=False)

    # combined_df から特定のカラムのみを含む新しい DataFrame を作成
    selected_columns_df = combined_df[['buyer_name', 'range_name', 'seller_name', 'message']]

    # DataFrame を JSON に変換して応答
    return selected_columns_df.to_dict(orient="records")

    # else:
    #     db.close()
    #     return JSONResponse(content={"deal_details": "購買履歴がありません"}, status_code=404)


