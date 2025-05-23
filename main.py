from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd

# 🎯 期待される回答例（教師なし）
expected_answers = [
    "ndiyo, watoto huenda shule",  # はい、子どもたちは学校へ行きます / Yes, children go to school
    "kila siku wanafunzi husoma",  # 毎日、生徒たちは勉強します / Students study every day
    "shule iko wazi kila siku",  # 学校は毎日開いています / The school is open every day
    "kila asubuhi wanafunzi husoma",  # 毎朝、生徒は勉強します / Every morning, students study
    "watoto wanaenda shule kila asubuhi",  # 子どもたちは毎朝学校に行きます / Children go to school every morning
    "mwanafunzi husoma kila siku",  # 生徒は毎日勉強します / A student studies every day
    "shule inaanza saa mbili asubuhi",  # 学校は朝8時に始まります / School starts at 8 a.m.
    "watoto wanasoma kwa bidii",  # 子どもたちは一生懸命勉強しています / Children study hard
    "wanafunzi huingia darasani mapema",  # 生徒たちは早く教室に入ります / Students enter the class early
    "shule hufunguliwa kila wiki",  # 学校は毎週開いています / The school opens every week
    "kila siku watoto huenda darasani",  # 毎日、子どもたちは教室に行きます / Children go to class every day
    "mwanafunzi huamka mapema kwenda shule",  # 生徒は学校に行くために早く起きます / Student wakes up early to go to school
    "wanafunzi huhudhuria masomo kila siku",  # 生徒たちは毎日授業に出席します / Students attend classes daily
    "shule hufunguliwa kila asubuhi",  # 学校は毎朝開きます / The school opens every morning
    "ndiyo, wanafunzi husoma kila siku",  # はい、生徒たちは毎日勉強します / Yes, students study daily
    "shule huanza kila asubuhi",  # 学校は毎朝始まります / School begins every morning
    "wanafunzi huenda shule kwa wakati",  # 生徒たちは時間通りに学校へ行きます / Students go to school on time
    "shule ipo wazi wakati wa masomo",  # 授業時間中は学校が開いています / School is open during lessons
    "ndiyo, watoto huingia shule kila siku",  # はい、子どもたちは毎日学校に入ります / Yes, children enter school daily
    "watoto husoma kila siku shuleni",  # 子どもたちは毎日学校で勉強します / Children study at school every day
    "watoto huamka mapema kwenda shule",  # 子どもたちは学校へ行くために早起きします / Children wake up early to go to school
    "shule hufunguliwa kila siku asubuhi",  # 学校は毎朝開きます / The school opens every morning
    "wanafunzi huenda darasani bila kuchelewa",  # 生徒たちは遅れずに教室へ行きます / Students go to class on time
    "masomo huanza kila asubuhi",  # 授業は毎朝始まります / Lessons begin every morning
    "ndiyo, mwanafunzi husoma kila siku",  # はい、生徒は毎日勉強します / Yes, the student studies daily
    "shule inaendelea kama kawaida",  # 学校は通常通り続いています / School is continuing as usual
    "watoto hushiriki masomo kila siku",  # 子どもたちは毎日授業に参加しています / Children participate in lessons daily
    "wanafunzi huandaliwa kwa mitihani",  # 生徒たちは試験の準備をしています / Students are prepared for exams
    "ndiyo, shule iko wazi leo",  # はい、今日は学校が開いています / Yes, the school is open today
    "watoto hupelekwa shule na wazazi",  # 子どもたちは親に学校まで送られます / Children are taken to school by parents
    "mwanafunzi husoma nyumbani na shuleni",  # 生徒は自宅でも学校でも勉強します / Student studies at home and at school
    "wanafunzi wanahudhuria vipindi vya asubuhi",  # 生徒は朝の授業に出席しています / Students attend morning classes
    "shule huanza saa mbili kila siku",  # 学校は毎日朝8時に始まります / School starts at 8 a.m. every day
    "watoto husoma somo la hisabati kila siku",  # 子どもたちは毎日算数を勉強します / Children study math daily
    "wanafunzi hukariri masomo yao nyumbani",  # 生徒たちは自宅で復習します / Students review their lessons at home
    "shule huendeshwa kwa utaratibu maalum",  # 学校は特別なスケジュールで運営されています / The school operates on a special schedule
    "kila mwanafunzi ana vitabu vya kiada",  # すべての生徒が教科書を持っています / Every student has textbooks
    "shule inafanya kazi siku tano kwa wiki",  # 学校は週5日運営されています / School runs five days a week
    "watoto hupokea chakula shuleni",  # 子どもたちは学校で食事を受け取ります / Children receive meals at school
    "wanafunzi husoma kwa kutumia kompyuta",  # 生徒たちはコンピュータを使って勉強しています / Students study using computers
    "ndiyo, mwanafunzi huvaa sare kila siku",  # はい、生徒は毎日制服を着ます / Yes, students wear uniforms daily
    "shule ina walimu wa kutosha",  # 学校には十分な教師がいます / The school has enough teachers
    "wanafunzi hupata mapumziko mchana",  # 生徒たちは昼に休憩をとります / Students have a lunch break
    "shule ina vifaa vya kisasa vya kufundishia",  # 学校には現代的な教材があります / School has modern teaching tools
    "mwanafunzi husoma kwa bidii nyumbani",  # 生徒は家でも一生懸命勉強しています / Students study hard at home
    "watoto hupenda kwenda shule kila siku",  # 子どもたちは毎日学校へ行くのが好きです / Children enjoy going to school daily
    "wanafunzi hupewa kazi za nyumbani kila siku",  # 生徒たちは毎日宿題を与えられます / Students are assigned homework daily
    "shule hutoa elimu bora kwa wanafunzi",  # 学校は生徒に質の高い教育を提供しています / School provides quality education
    "ndiyo, shule ni muhimu kwa watoto",  # はい、学校は子どもたちにとって重要です / Yes, school is important for children
    "wanafunzi hukusanyika kwenye uwanja kabla ya darasa",  # 生徒たちは授業前に校庭に集まります / Students gather in the field before class
    "shule ni ya maana kwa watoto",  # 学校は子どもにとって意味がある / School is meaningful for children
    "elimu ni muhimu kwa watoto",  # 教育は子どもにとって重要 / Education is important for children
    "watoto wanahitaji shule kwa maendeleo yao",  # 子どもは成長のために学校が必要 / Children need school for development
    "shule ni sehemu muhimu ya maisha ya watoto",  # 学校は子どもの生活の重要な一部 / School is an important part of children’s lives
    "elimu ni msingi wa maisha ya baadaye",  # 教育は将来の基盤です / Education is the foundation of future life
    "watoto wanahitaji elimu ili kufanikiwa",  # 子どもは成功のために教育が必要です / Children need education to succeed
    "shule huwasaidia watoto kujifunza na kukua",  # 学校は子どもが学び成長するのを助けます / School helps children learn and grow
    "bila elimu watoto hukosa fursa muhimu",  # 教育がなければ子どもは重要な機会を逃します / Without education, children miss key opportunities
    "elimu hufungua milango ya mafanikio",  # 教育は成功への扉を開きます / Education opens doors to success
    "shule ni mahali pa kujenga mustakabali bora",  # 学校はよりよい未来を築く場所です / School builds a better future
    "kupata elimu ni haki ya kila mtoto",  # 教育を受けることはすべての子どもの権利です / Education is every child’s right
    "shule huwapa watoto ujuzi wa maisha",  # 学校は子どもに生活スキルを与えます / School gives life skills to children    
    "elimu huimarisha uwezo wa kufikiri kwa watoto",       # 教育は子どもの思考力を高めます / Education enhances children's thinking ability
    "watoto wenye elimu wana nafasi nzuri zaidi maishani", # 教育のある子どもはより良い人生のチャンスがあります / Educated children have better life chances
    "masomo ya shule",  # 学校の教科 / School subjects
    "ratiba ya shule",  # 学校の時間割 / School schedule
    "mwanafunzi anasoma",  # 生徒が勉強している / A student is studying
    "kitabu cha historia",  # 歴史の本 / History book
    "mtihani utaanza kesho",  # 試験は明日始まる / Exam starts tomorrow
    "walimu wanasaidia wanafunzi",  # 教師が生徒を助けている / Teachers help students
    "darasa limejaa wanafunzi",  # 教室は生徒でいっぱい / Classroom is full of students
    "mwalimu anafundisha hesabu",  # 先生が算数を教えている / Teacher is teaching math
    "shule ya msingi",  # 小学校 / Primary school
    "wanafunzi wanacheza uwanjani",  # 生徒が校庭で遊んでいる / Students are playing in the field
    "vitabu vya kiada vinatolewa",  # 教科書が配布されている / Textbooks are being distributed
    "kalenda ya masomo ya mwaka huu",  # 今年の学習カレンダー / Academic calendar for this year
    "mwanafunzi amepewa kazi ya nyumbani",  # 生徒が宿題を与えられた / Student got homework
    "wanafunzi wanasoma kwa bidii",  # 生徒たちは一生懸命勉強している / Students study hard
    "walimu wanafanya mkutano",  # 教師が会議をしている / Teachers are having a meeting
    "ratiba mpya ya masomo imetolewa",  # 新しい時間割が発表された / New class schedule released
    "wanafunzi wamehitimu darasa la saba",  # 生徒が7年生を修了した / Students graduated from 7th grade
    "shule imefungwa kwa likizo",  # 学校は休暇で閉まっている / School closed for holiday
    "vitabu vya sayansi vimewasili",  # 理科の本が届いた / Science books have arrived
    "walimu wapya wameajiriwa",  # 新しい教師が雇われた / New teachers have been hired
    "shule imekarabatiwa",  # 学校が改修された / The school has been renovated
    "wanafunzi walifanya mitihani",  # 生徒が試験を受けた / Students took exams
    "mwalimu mkuu alihutubia wanafunzi",  # 校長が生徒に演説した / Principal addressed the students
    "walimu walihudhuria mafunzo",  # 教師が研修に参加した / Teachers attended training
    "mwanafunzi alipata tuzo",  # 生徒が賞を受け取った / A student received an award
    "darasa lina viti vipya",  # 教室には新しい椅子がある / Classroom has new chairs
    "wanafunzi walifanya utafiti",  # 生徒が調査を行った / Students did research
    "shule ina maktaba kubwa",  # 学校には大きな図書館がある / School has a large library
    "ratiba ya likizo imetolewa",  # 休暇の予定が出された / Holiday schedule released
    "wanafunzi walipokea sare mpya",  # 生徒たちは新しい制服を受け取った / Students received new uniforms
    "elimu bora hujenga taifa imara",  # 質の高い教育は強い国家を築く / Quality education builds a strong nation
    "watoto walioelimika wanaweza kuchangia jamii",  # 教育を受けた子どもは社会に貢献できる / Educated children contribute to society
    "shule hufundisha nidhamu na maadili mema",  # 学校は規律と良い価値観を教える / Schools teach discipline and values
    "kupata elimu ni hatua ya kwanza ya mafanikio",  # 教育を受けることは成功への第一歩 / Education is the first step to success
    "watoto wasio na elimu hukabili changamoto nyingi",  # 教育のない子どもは多くの困難に直面する / Uneducated children face many challenges
    "elimu huongeza uelewa wa ulimwengu",  # 教育は世界の理解を深める / Education increases global understanding
    "shule ni chanzo cha maarifa kwa watoto",  # 学校は子どもにとって知識の源 / School is a source of knowledge
    "watoto wanapopata elimu, jamii inanufaika",  # 子どもが教育を受けると社会全体が利益を得る / Society benefits when children are educated
    "elimu huwasaidia watoto kuwa huru kifikra",  # 教育は子どもが自由に考える力を養う / Education fosters independent thinking
    "shule inawajenga watoto kuwa viongozi wa baadaye",  # 学校は将来のリーダーを育てる / Schools build future leaders
    "kupitia elimu watoto hujifunza kushirikiana na wengine",  # 教育を通じて子どもは協力の大切さを学ぶ / Children learn collaboration through education
    "elimu ni nguzo muhimu ya maendeleo ya binadamu",  # 教育は人間の発展の重要な柱 / Education is a pillar of human development
    "shule huandaa watoto kwa maisha ya baadaye",  # 学校は子どもを将来に備えさせる / Schools prepare children for life
    "elimu hujenga imani na kujiamini kwa mtoto",  # 教育は自信と信頼を育てる / Education builds trust and confidence
    "watoto walioelimika hupata fursa nyingi zaidi",  # 教育を受けた子どもはより多くの機会を得る / Educated children get more opportunities
    "shule hutoa mazingira salama ya kujifunza",  # 学校は安全な学びの場を提供する / School provides a safe learning environment
    "elimu huwapa watoto sauti katika jamii",  # 教育は子どもに社会での声を与える / Education gives children a voice
    "shule ni msingi wa maendeleo ya kiuchumi ya jamii",  # 学校は社会の経済発展の基礎 / School is the foundation of economic development
    "elimu huondoa ujinga na kuleta mwanga",  # 教育は無知をなくし、光をもたらす / Education removes ignorance and brings light
    "kupata elimu ni uwekezaji bora kwa maisha ya mtoto"  # 教育は子どもの人生への最良の投資 / Education is the best investment in a child's life
]

# 📝 ユーザーの自由回答（Whisperなどの音声認識結果）
user_answer = "wanahitaji shule kwa"

# 🧮 TF-IDFベクトル化
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

X = vectorizer.fit_transform(expected_answers + [user_answer])

# 📈 全スコア表示
similarities = cosine_similarity(X[-1], X[:-1]).flatten()
ranked = sorted(zip(similarities, expected_answers), reverse=True)

for i, (sim, ref) in enumerate(ranked[:15], 1):  # 上位5件を表示
    print(f"{i}. Score: {sim:.3f} → \"{ref}\"")


import json

tfidf_data = {
    "vocabulary": vectorizer.vocabulary_,
    "idf": vectorizer.idf_.tolist(),
    "expected_answers": expected_answers
}
with open("tfidf_model.json", "w", encoding="utf-8") as f:
    json.dump(tfidf_data, f, ensure_ascii=False, indent=2)
