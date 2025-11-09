from datetime import datetime
from sqlalchemy import Column, Integer, Text, String, Boolean, TIMESTAMP, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    object_key = Column(String, unique=True, nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    # связь с items
    item = relationship("Item", back_populates="image", uselist=False)

class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    description = Column(Text)
    image_key = Column(String, ForeignKey("images.object_key"), nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    is_deleted = Column(Boolean, default=False)

    image = relationship("Image", back_populates="item")
